import tensorflow as tf
import h5py, os, yaml
import numpy as np
import re
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import tfr_evaluate, utils
import explain
import embed
from tqdm import tqdm
from dinuc_shuffle import dinuc_shuffle
import itertools

class GlobalImportance():
    """Class that performs GIA experiments."""
    def __init__(self, model, targets, alphabet='ACGT'):
        self.model = model
        self.alphabet = alphabet
        self.x_null = None
        self.x_null_index = None
        self.embedded_predictions = {}
        self.seqs_with = {}
        self.seqs_removed = {}
        self.summary_remove_motifs = []
        self.seq_idx = {}
        self.targets = targets

    # methods for removing motifs
    def set_seqs_for_removing(self, subset, num_sample, seed):
        if num_sample:
            print('SUBSETTING SEQUENCES')
            if seed:
                np.random.seed(seed)
            rand_idx = np.random.choice(subset.shape[0], num_sample, replace=False).flatten()
            self.seqs_to_remove_motif = subset[rand_idx]
        else:
            self.seqs_to_remove_motif = subset

    def occlude_all_motif_instances(self, subset, tandem_motifs_to_remove,
                                    num_sample=None,
                                    seed=42, func='max', batch_size=32):
        self.set_seqs_for_removing(subset, num_sample, seed)
        print('tandem_motifs_to_remove', tandem_motifs_to_remove)
        motif_key = ', '.join(tandem_motifs_to_remove)
        print(motif_key)
        self.seqs_with[motif_key], self.seqs_removed[motif_key], self.n_instances, self.seq_idx[motif_key] = randomize_multiple_seqs(self.seqs_to_remove_motif,
                                                      tandem_motifs_to_remove, self.model, window_size=None)
        if len(self.seqs_with[motif_key]) > 0:
            self.seqs_with[motif_key], self.seqs_removed[motif_key] = [np.array(n) for n in [self.seqs_with[motif_key], self.seqs_removed[motif_key]]]
            df = self.get_predictions(motif_key, batch_size, func)
        else:
            print('No seqs detected')
            df = pd.DataFrame({func+' coverage':[None], 'sequence':[None], 'N instances':[None], 'motif pattern': [motif_key]})
        self.summary_remove_motifs.append(df)

    def get_predictions(self, motif_key, batch_size, func):
        ori_preds = embed.predict_np((self.seqs_with[motif_key]),
                                    self.model, batch_size=batch_size,
                                    reshape_to_2D=False)

        del_preds = get_avg_preds(self.seqs_removed[motif_key],
                                                    self.model)
        if ori_preds.ndim == 2 and del_preds.ndim == 2:
            print('Reshaping to 3D!')
            ori_preds = np.expand_dims(ori_preds, axis=1)
            del_preds = np.expand_dims(del_preds, axis=1)
        print(ori_preds.shape)
        max_ori_pc3 = np.mean(make_3D(ori_preds), axis=1) #eval('np.'+func)(make_3D(ori_preds), axis=1)
        max_pred_pc3 = np.mean(make_3D(del_preds), axis=1) #eval('np.'+func)(make_3D(del_preds), axis=1)
        df_all = pd.DataFrame({
                                'mean coverage': np.concatenate([max_ori_pc3.flatten(), max_pred_pc3.flatten()]),
                                'sequence':['original' for i in range(len(max_ori_pc3.flatten()))]+['removed' for i in range(len(max_pred_pc3.flatten()))],
                                'cell line': np.concatenate([np.tile(self.targets, max_ori_pc3.shape[0]) for i in range(2)]),
                                'N instances':np.concatenate([np.repeat(self.n_instances, len(self.targets)) for i in range(2)]),
                                'seq_idx': np.concatenate([np.repeat(self.seq_idx[motif_key], len(self.targets)) for i in range(2)])
                                })
        df_all['motif pattern'] = motif_key
        return df_all



    def set_null_model(self, null_model, base_sequence, num_sample=1000,
                        binding_scores=None, seed=None):
        """use model-based approach to set the null sequences"""
        self.x_null = generate_null_sequence_set(null_model, base_sequence, num_sample, binding_scores, seed)
        self.x_null_index = np.argmax(self.x_null, axis=2)
        self.predict_null()


    def set_x_null(self, x_null):
        """set the null sequences"""
        self.x_null = x_null
        self.x_null_index = np.argmax(x_null, axis=2)
        self.predict_null()


    def predict_null(self):
        """perform GIA on null sequences"""
        self.null_profiles = self.model.predict(self.x_null)


    def embed_patterns(self, patterns):
        """embed patterns in null sequences"""
        if not isinstance(patterns, list):
            patterns = [patterns]

        x_index = np.copy(self.x_null_index)
        for pattern, position in patterns:

            # convert pattern to categorical representation
            pattern_index = np.array([self.alphabet.index(i) for i in pattern])

            # embed pattern
            x_index[:,position:position+len(pattern)] = pattern_index

        # convert to categorical representation to one-hot
        one_hot = np.zeros((len(x_index), len(x_index[0]), len(self.alphabet)))
        for n, x in enumerate(x_index):
            for l, a in enumerate(x):
                one_hot[n,l,a] = 1.0

        return one_hot

    def embed_predict_quant_effect(self, patterns):
        """embed pattern in null sequences and get their predictions"""
        one_hot = self.embed_patterns(patterns)
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m,p in patterns])
        self.embedded_predictions[pattern_label] = self.model.predict(one_hot)
        assert self.embedded_predictions[pattern_label].shape == self.null_profiles.shape
        if self.embedded_predictions[pattern_label].ndim == 2:
            return np.expand_dims(self.embedded_predictions[pattern_label] - self.null_profiles, axis=1)
        else:
            return self.embedded_predictions[pattern_label] - self.null_profiles

    def positional_bias(self, motif, positions, targets):
        """GIA to find positional bias"""
        # loop over positions and measure effect size of intervention
        all_scores = []
        for position in tqdm(positions):
            all_scores.append(self.embed_predict_quant_effect([(motif, position)]))
        mean_per_pos = np.array(all_scores).mean(axis=1).mean(axis=1)
        df = pd.DataFrame({'position':np.repeat(positions, len(targets)),
                            'mean difference':np.array(mean_per_pos).flatten(),
                            'cell line': np.tile(targets, mean_per_pos.shape[0])})
        df['motif'] = motif
        return df

    def multiple_sites(self, motif, positions):
        """GIA to find relation with multiple binding sites"""

        # loop over positions and measure effect size of intervention
        all_scores = []
        for i, position in enumerate(positions):
            # embed motif multiple times
            interventions = []
            for j in range(i+1):
                interventions.append((motif, positions[j]))
            all_scores.append(self.embed_predict_quant_effect(interventions))
        return np.array(all_scores)


#-------------------------------------------------------------------------------------
# Null sequence models
#-------------------------------------------------------------------------------------
def generate_null_sequence_set(null_model, base_sequence, num_sample=1000 , binding_scores=None, seed=None):
    if null_model == 'random':    return generate_shuffled_set(base_sequence, num_sample)
    if null_model == 'profile':   return generate_profile_set(base_sequence, num_sample)
    if null_model == 'dinuc':     return generate_dinucleotide_shuffled_set(base_sequence, num_sample)
    if null_model == 'quartile1': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=1)
    if null_model == 'quartile2': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=2)
    if null_model == 'quartile3': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=3)
    if null_model == 'quartile4': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=4)
    if null_model == 'none':
        if seed:
            np.random.seed(seed)
            print('seed set!')
        idx = np.random.choice(base_sequence.shape[0], num_sample)

        return base_sequence[idx]
    else: print ('null_model name not recognized.')


def generate_profile_set(base_sequence, num_sample):
    # set null sequence model
    seq_model = np.mean(np.squeeze(base_sequence), axis=0)
    seq_model /= np.sum(seq_model, axis=1, keepdims=True)

    # sequence length
    L = seq_model.shape[0]

    x_null = np.zeros((num_sample, L, 4))
    for n in range(num_sample):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0,1,L)

        # calculate cumulative sum of the probabilities
        cum_prob = seq_model.cumsum(axis=1)

        # find bin that matches random number for each position
        for l in range(L):
            index = [j for j in range(4) if Z[l] < cum_prob[l,j]][0]
            x_null[n,l,index] = 1
    return x_null


def generate_shuffled_set(base_sequence, num_sample):
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle nucleotides
    [np.random.shuffle(x) for x in x_null]
    return x_null


def generate_dinucleotide_shuffled_set(base_sequence, num_sample):

    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle dinucleotides
    for j, seq in enumerate(x_null):
        x_null[j] = dinuc_shuffle(seq)
    return x_null


def generate_quartile_set(base_sequence, num_sample, binding_scores, quartile):
    # sort sequences by the binding score (descending order)
    sort_index = np.argsort(binding_scores[:,0])[::-1]
    base_sequence = base_sequence[sort_index]

    # set quartile indices
    L = len(base_sequence)
    L0, L1, L2, L3, L4 = [0, int(L/4), int(L*2/4), int(L*3/4), L]

    # pick the quartile:
    if (quartile==1): base_sequence = base_sequence[L0:L1]
    if (quartile==2): base_sequence = base_sequence[L1:L2]
    if (quartile==3): base_sequence = base_sequence[L2:L3]
    if (quartile==4): base_sequence = base_sequence[L3:L4]

    # now shuffle the sequences
    shuffle = np.random.permutation(len(base_sequence))

    # take a smaller sample of size num_sample
    return base_sequence[shuffle[:num_sample]]


#-------------------------------------------------------------------------------------
# util functions
#-------------------------------------------------------------------------------------
def select_set(testset_type, C, X, Y, cell_line=None):
    if testset_type == 'whole':
        return (X)
    elif testset_type == 'all_threshold':
        threshold_mask = (Y.max(axis=1)>2).any(axis=-1)
        return (X[threshold_mask])
    elif testset_type == 'cell_high':
        assert cell_line, 'No cell line provided!'
        _, thresh_X, _ = embed.threshold_cell_line_np(C, X, Y, cell_line,
                                                    more_than=2,
                                                    less_than=None)
        return (thresh_X)
    elif testset_type == 'cell_low':
        assert cell_line, 'No cell line provided!'
        _, thresh_X, _ = embed.threshold_cell_line_np(C, X, Y, cell_line,
                                                    more_than=1,
                                                    less_than=2)
        return (thresh_X)
    else:
        print('Wrong please try again thank you bye')
        exit()

def make_3D(array):
    if len(array.shape)==2:
        return np.expand_dims(array, axis=0)
    elif len(array.shape) == 3:
        return array
    else:
        print('bad array')
        exit()

def boxplot_with_test(data, x, y, pairs):
    plotting_parameters = {
                            'data':    data,
                            'x':       x,
                            'y':       y}
    pvalues = [mannwhitneyu(data[data[x]==pair[0]][y],
                            data[data[x]==pair[1]][y]).pvalue for pair in pairs]
    ax = sns.boxplot(**plotting_parameters)
    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.set_pvalues(pvalues)
    annotator.annotate();

#-------------------------------------------------------------------------------------
# functions to find a motif in a sequence
#-------------------------------------------------------------------------------------
def find_motif_indices(motif_pattern, str_seq):
    '''Find all str motif start positions in a sequence str'''
    iter = re.finditer(motif_pattern, str_seq)
    return [m.start(0) for m in iter]

def find_max_saliency_ind(indices, saliency_values):
    '''find motif instance closest to the max saliency value'''
    max_point = np.argmax(saliency_values)
    if len(indices)>0:
        return [indices[np.abs(indices-max_point).argmin()]]
    else:
        return []


def filter_indices_in_saliency_peak(indices, saliency_values, window=300):
    '''filter motifs within a window around the max saliency'''
    max_point = np.argmax(saliency_values)
    if len(indices)>0:
        return list(np.array(indices)[(np.abs(indices-max_point)<window/2)])
    else:
        return []


def select_indices(motif_pattern, str_seq, saliency_values=None,
                   max_only=False, filter_window=False):
    '''select indices according to filtering criteria'''
    indices = find_motif_indices(motif_pattern, str_seq)
    if max_only:
        return find_max_saliency_ind(indices, saliency_values)
    elif filter_window:
        return filter_indices_in_saliency_peak(indices, saliency_values, filter_window)
    else: # find all
        return indices


def find_multiple_motifs(motif_pattern_list, str_seq, saliency_values=None,
                        max_only=False, filter_window=False):
    '''find indices of multiple motifs in a single sequence'''
    motifs_and_indices = {}
    for motif_pattern in motif_pattern_list:
        chosen_ind = select_indices(motif_pattern, str_seq,
                                    saliency_values,
                                    max_only, filter_window )
        motifs_and_indices[motif_pattern] = chosen_ind
    return motifs_and_indices


#-------------------------------------------------------------------------------------
# functions to remove or randomize a motif
#-------------------------------------------------------------------------------------

def randomize_motif_dict_in_seq(motifs_and_indices, selected_seq, n_occlusions=25):
    modified_seqs = []
    for i in range(n_occlusions):
        modified_seq = selected_seq.copy()
        for motif_pattern, motif_start_indices in motifs_and_indices.items():
            for motif_start in motif_start_indices:
                random_pattern = np.array([[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]][np.random.randint(4)] for i in range(len(motif_pattern))])
                modified_seq[motif_start:motif_start+len(motif_pattern)] = random_pattern
        modified_seqs.append(modified_seq)
    return np.array(modified_seqs)


def randomize_multiple_seqs(onehot_seqs, tandem_motifs_to_remove, model,
                            cell_line=None, window_size=None):
    seqs_with_motif = []
    seqs_removed_motifs = []
    # saliency_all_seqs = explain.get_multiple_saliency_values(onehot_seqs, model, cell_line)
    n_instances_per_seq = []
    incl_idx = []
    for o, onehot_seq in tqdm(enumerate(onehot_seqs)):
        str_seq = ''.join(util.onehot_to_str(onehot_seq))
        motifs_and_indices = find_multiple_motifs(tandem_motifs_to_remove, str_seq,
                             # saliency_values=saliency_all_seqs[o],
                             filter_window=window_size)
        all_motifs_present = np.array([len(v)>0 for k,v in motifs_and_indices.items()]).all()
        if all_motifs_present:
            seqs_with_motif.append(onehot_seq.copy())
            seqs_removed_motifs.append(randomize_motif_dict_in_seq(motifs_and_indices,
                                                              onehot_seq))
            n_instances_per_seq.append([str(len(v)) for k,v in motifs_and_indices.items()])
            incl_idx.append(o)
    n_instances_per_seq = [', '.join(n) for n in n_instances_per_seq]
    return (seqs_with_motif, seqs_removed_motifs, n_instances_per_seq, incl_idx)


def get_avg_preds(seqs_removed, model):
    N,B,L,C = seqs_removed.shape
    removed_preds = embed.predict_np((seqs_removed.reshape(N*B,L,C)), model,
                                     batch_size=32, reshape_to_2D=False)#[:,:,cell_line]
    print(removed_preds.shape)
    if removed_preds.ndim == 2:
        _,C = removed_preds.shape
        avg_removed_preds = removed_preds.reshape(N,B,C).mean(axis=1)
    elif removed_preds.ndim == 3:
        _,L,C = removed_preds.shape
        avg_removed_preds = removed_preds.reshape(N,B,L,C).mean(axis=1)
    else:
        sys.exit('Unsupported prediction shape')
    return avg_removed_preds

def test_flanks(gi, all_flanks, targets, position=1024, output_path=''):
    all_scores = []
    for motif in tqdm(all_flanks):
        diff_scores = gi.embed_predict_quant_effect([(motif, position)])
        # if diff_scores.ndim == 2:
        #     diff_scores = np.expand_dims(diff_scores, axis=1)
        all_scores_per_motif=(diff_scores).mean(axis=0).mean(axis=0)
        all_scores.append(all_scores_per_motif)

    df = pd.DataFrame({'motif':np.repeat(all_flanks, len(targets)),
                  'mean difference':np.array(all_scores).flatten(),
                 'cell line': np.tile(targets, len(all_flanks))})
    df.to_csv(output_path, index=None)
    return df


def generate_flanks(motif_pattern):
    dot_positions = np.argwhere(np.array(list(motif_pattern))=='.').flatten()
    kmer_size = len(dot_positions)
    kmers = ["".join(p) for p in itertools.product(list('ACGT'), repeat=kmer_size)]
    all_motifs = []
    for kmer in tqdm(kmers):
        motif_with_flanking_nucls = list(motif_pattern)
        for p, pos in enumerate(dot_positions):
            motif_with_flanking_nucls[pos] = kmer[p]
        motif_with_flanking_nucls = ''.join(motif_with_flanking_nucls)
        all_motifs.append(motif_with_flanking_nucls)
    return all_motifs
