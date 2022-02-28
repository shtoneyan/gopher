import h5py
import itertools
import numpy as np
import os
import pandas as pd
import evaluate
import re
import seaborn as sns
import tensorflow as tf
import utils
import yaml
from dinuc_shuffle import dinuc_shuffle
from scipy import stats
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from tqdm import tqdm


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
        """
        select sequences to remove a given motif from
        :param subset: a set of onehot sequences in np array
        :param num_sample: number to limit the sequences to
        :param seed: random sample seed
        :return:
        """
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
        self.seqs_with[motif_key], self.seqs_removed[motif_key], self.n_instances, self.seq_idx[
            motif_key] = randomize_multiple_seqs(self.seqs_to_remove_motif,
                                                 tandem_motifs_to_remove, self.model, window_size=None)
        if len(self.seqs_with[motif_key]) > 0:
            self.seqs_with[motif_key], self.seqs_removed[motif_key] = [np.array(n) for n in [self.seqs_with[motif_key],
                                                                                             self.seqs_removed[
                                                                                                 motif_key]]]
            df = self.get_predictions(motif_key, batch_size, func)
        else:
            print('No seqs detected')
            df = pd.DataFrame(
                {func + ' coverage': [None], 'sequence': [None], 'N instances': [None], 'motif pattern': [motif_key]})
        self.summary_remove_motifs.append(df)

    def get_predictions(self, motif_key, batch_size, func):
        ori_preds = utils.predict_np((self.seqs_with[motif_key]),
                                     self.model, batch_size=batch_size,
                                     reshape_to_2D=False)

        del_preds = get_avg_preds(self.seqs_removed[motif_key],
                                  self.model)
        if ori_preds.ndim == 2 and del_preds.ndim == 2:
            print('Reshaping to 3D!')
            ori_preds = np.expand_dims(ori_preds, axis=1)
            del_preds = np.expand_dims(del_preds, axis=1)
        print(ori_preds.shape)
        max_ori_pc3 = np.mean(make_3D(ori_preds), axis=1)  # eval('np.'+func)(make_3D(ori_preds), axis=1)
        max_pred_pc3 = np.mean(make_3D(del_preds), axis=1)  # eval('np.'+func)(make_3D(del_preds), axis=1)
        df_all = pd.DataFrame({
            'mean coverage': np.concatenate([max_ori_pc3.flatten(), max_pred_pc3.flatten()]),
            'sequence': ['original' for i in range(len(max_ori_pc3.flatten()))] + ['removed' for i in
                                                                                   range(len(max_pred_pc3.flatten()))],
            'cell line': np.concatenate([np.tile(self.targets, max_ori_pc3.shape[0]) for i in range(2)]),
            'N instances': np.concatenate([np.repeat(self.n_instances, len(self.targets)) for i in range(2)]),
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
            x_index[:, position:position + len(pattern)] = pattern_index

        # convert to categorical representation to one-hot
        one_hot = np.zeros((len(x_index), len(x_index[0]), len(self.alphabet)))
        for n, x in enumerate(x_index):
            for l, a in enumerate(x):
                one_hot[n, l, a] = 1.0

        return one_hot

    def embed_predict_quant_effect(self, patterns):
        """embed pattern in null sequences and get their predictions"""
        one_hot = self.embed_patterns(patterns)
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m, p in patterns])
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
        df = pd.DataFrame({'position': np.repeat(positions, len(targets)),
                           'mean difference': np.array(mean_per_pos).flatten(),
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
            for j in range(i + 1):
                interventions.append((motif, positions[j]))
            all_scores.append(self.embed_predict_quant_effect(interventions))
        return np.array(all_scores)


# -------------------------------------------------------------------------------------
# Null sequence models
# -------------------------------------------------------------------------------------
def generate_null_sequence_set(null_model, base_sequence, num_sample=1000, binding_scores=None, seed=None):
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
    else:
        print ('null_model name not recognized.')


def generate_profile_set(base_sequence, num_sample):
    # set null sequence model
    seq_model = np.mean(np.squeeze(base_sequence), axis=0)
    seq_model /= np.sum(seq_model, axis=1, keepdims=True)

    # sequence length
    L = seq_model.shape[0]

    x_null = np.zeros((num_sample, L, 4))
    for n in range(num_sample):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0, 1, L)

        # calculate cumulative sum of the probabilities
        cum_prob = seq_model.cumsum(axis=1)

        # find bin that matches random number for each position
        for l in range(L):
            index = [j for j in range(4) if Z[l] < cum_prob[l, j]][0]
            x_null[n, l, index] = 1
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
    sort_index = np.argsort(binding_scores[:, 0])[::-1]
    base_sequence = base_sequence[sort_index]

    # set quartile indices
    L = len(base_sequence)
    L0, L1, L2, L3, L4 = [0, int(L / 4), int(L * 2 / 4), int(L * 3 / 4), L]

    # pick the quartile:
    if (quartile == 1): base_sequence = base_sequence[L0:L1]
    if (quartile == 2): base_sequence = base_sequence[L1:L2]
    if (quartile == 3): base_sequence = base_sequence[L2:L3]
    if (quartile == 4): base_sequence = base_sequence[L3:L4]

    # now shuffle the sequences
    shuffle = np.random.permutation(len(base_sequence))

    # take a smaller sample of size num_sample
    return base_sequence[shuffle[:num_sample]]


# -------------------------------------------------------------------------------------
# util functions
# -------------------------------------------------------------------------------------
def select_set(testset_type, C, X, Y, cell_line=None):
    """
    This function selects sequences for constructing background sequences
    :param testset_type:
    :param C:
    :param X:
    :param Y:
    :param cell_line: cell line according to which to filter if
    :return: numpy array of selected sequences
    """
    if testset_type == 'all_threshold':
        threshold_mask = (Y.max(axis=1) > 2).any(axis=-1)
        return X[threshold_mask]
    elif testset_type == 'cell_low':
        assert cell_line, 'No cell line provided!'
        _, thresh_X, _ = utils.threshold_cell_line_np(C, X, Y, cell_line,
                                                      more_than=1,
                                                      less_than=2)
        return thresh_X
    else:
        print('Wrong please try again thank you bye')
        exit()


def make_3D(array):
    if len(array.shape) == 2:
        return np.expand_dims(array, axis=0)
    elif len(array.shape) == 3:
        return array
    else:
        print('bad array')
        exit()


def boxplot_with_test(data, x, y, pairs):
    plotting_parameters = {
        'data': data,
        'x': x,
        'y': y}
    pvalues = [mannwhitneyu(data[data[x] == pair[0]][y],
                            data[data[x] == pair[1]][y]).pvalue for pair in pairs]
    ax = sns.boxplot(**plotting_parameters)
    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.set_pvalues(pvalues)
    annotator.annotate();


# -------------------------------------------------------------------------------------
# functions to find a motif in a sequence
# -------------------------------------------------------------------------------------
def find_motif_indices(motif_pattern, str_seq):
    '''Find all str motif start positions in a sequence str'''
    iter = re.finditer(motif_pattern, str_seq)
    return [m.start(0) for m in iter]


def find_max_saliency_ind(indices, saliency_values):
    '''find motif instance closest to the max saliency value'''
    max_point = np.argmax(saliency_values)
    if len(indices) > 0:
        return [indices[np.abs(indices - max_point).argmin()]]
    else:
        return []


def filter_indices_in_saliency_peak(indices, saliency_values, window=300):
    '''filter motifs within a window around the max saliency'''
    max_point = np.argmax(saliency_values)
    if len(indices) > 0:
        return list(np.array(indices)[(np.abs(indices - max_point) < window / 2)])
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
    else:  # find all
        return indices


def find_multiple_motifs(motif_pattern_list, str_seq, saliency_values=None,
                         max_only=False, filter_window=False):
    '''find indices of multiple motifs in a single sequence'''
    motifs_and_indices = {}
    for motif_pattern in motif_pattern_list:
        chosen_ind = select_indices(motif_pattern, str_seq,
                                    saliency_values,
                                    max_only, filter_window)
        motifs_and_indices[motif_pattern] = chosen_ind
    return motifs_and_indices


# -------------------------------------------------------------------------------------
# functions to remove or randomize a motif
# -------------------------------------------------------------------------------------

def randomize_motif_dict_in_seq(motifs_and_indices, selected_seq, n_occlusions=25):
    modified_seqs = []
    for i in range(n_occlusions):
        modified_seq = selected_seq.copy()
        for motif_pattern, motif_start_indices in motifs_and_indices.items():
            for motif_start in motif_start_indices:
                random_pattern = np.array(
                    [[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]][np.random.randint(4)] for
                     i in range(len(motif_pattern))])
                modified_seq[motif_start:motif_start + len(motif_pattern)] = random_pattern
        modified_seqs.append(modified_seq)
    return np.array(modified_seqs)


def randomize_multiple_seqs(onehot_seqs, tandem_motifs_to_remove, model,
                            cell_line=None, window_size=None):
    seqs_with_motif = []
    seqs_removed_motifs = []
    n_instances_per_seq = []
    incl_idx = []
    for o, onehot_seq in tqdm(enumerate(onehot_seqs)):
        str_seq = ''.join(utils.onehot_to_str(onehot_seq))
        motifs_and_indices = find_multiple_motifs(tandem_motifs_to_remove, str_seq,
                                                  # saliency_values=saliency_all_seqs[o],
                                                  filter_window=window_size)
        all_motifs_present = np.array([len(v) > 0 for k, v in motifs_and_indices.items()]).all()
        if all_motifs_present:
            seqs_with_motif.append(onehot_seq.copy())
            seqs_removed_motifs.append(randomize_motif_dict_in_seq(motifs_and_indices,
                                                                   onehot_seq))
            n_instances_per_seq.append([str(len(v)) for k, v in motifs_and_indices.items()])
            incl_idx.append(o)
    n_instances_per_seq = [', '.join(n) for n in n_instances_per_seq]
    return (seqs_with_motif, seqs_removed_motifs, n_instances_per_seq, incl_idx)


def get_avg_preds(seqs_removed, model):
    N, B, L, C = seqs_removed.shape
    removed_preds = utils.predict_np((seqs_removed.reshape(N * B, L, C)), model,
                                     batch_size=32, reshape_to_2D=False)  # [:,:,cell_line]
    print(removed_preds.shape)
    if removed_preds.ndim == 2:
        _, C = removed_preds.shape
        avg_removed_preds = removed_preds.reshape(N, B, C).mean(axis=1)
    elif removed_preds.ndim == 3:
        _, L, C = removed_preds.shape
        avg_removed_preds = removed_preds.reshape(N, B, L, C).mean(axis=1)
    else:
        sys.exit('Unsupported prediction shape')
    return avg_removed_preds


def test_flanks(gi, all_flanks, targets, position=1024, output_path=''):
    all_scores = []
    for motif in tqdm(all_flanks):
        diff_scores = gi.embed_predict_quant_effect([(motif, position)])
        # if diff_scores.ndim == 2:
        #     diff_scores = np.expand_dims(diff_scores, axis=1)
        all_scores_per_motif = (diff_scores).mean(axis=0).mean(axis=0)
        all_scores.append(all_scores_per_motif)

    df = pd.DataFrame({'motif': np.repeat(all_flanks, len(targets)),
                       'mean difference': np.array(all_scores).flatten(),
                       'cell line': np.tile(targets, len(all_flanks))})
    df.to_csv(output_path, index=None)
    return df


def generate_flanks(motif_pattern):
    dot_positions = np.argwhere(np.array(list(motif_pattern)) == '.').flatten()
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


def record_flank_test(gi, motif, targets, cell_line_name, flanks_path, motif1_positions=[1024]):
    # select the best flanks based on where the dots are in the pattern
    if '.' in motif:
        if os.path.isfile(flanks_path):
            flank_scores = pd.read_csv(flanks_path)
        else:
            all_motifs = generate_flanks(motif)
            print('Testing flanks')
            flank_scores = test_flanks(gi, all_motifs, targets,
                                       output_path=flanks_path)
        best_flank = flank_scores[flank_scores['cell line'] == cell_line_name].sort_values('mean difference').iloc[-1, 0]
        print(best_flank)
    else:
        best_flank = motif
    return best_flank



def optimize_distance(gi, optimized_motifs, targets, distance_path, first_motif_pos):
    if os.path.isfile(distance_path):
        df = pd.read_csv(distance_path)
    else:
        two_motifs_pos_scores = []
        positions = list(range(0,2048-len(optimized_motifs[1]),2))
        for position in tqdm(positions):
            diff_scores = gi.embed_predict_quant_effect([(optimized_motifs[0], first_motif_pos),
                                                        (optimized_motifs[1],
                                                            position)])
            two_motifs_pos_scores.append(diff_scores.mean(axis=0).mean(axis=0))
        two_motifs_pos_scores = np.array(two_motifs_pos_scores)
        motif_2_label = np.array(['{}_{}'.format(optimized_motifs[1], pos) for pos in positions])
        df = pd.DataFrame({
                            'motif 2':np.repeat(motif_2_label, len(targets)),
                            'mean difference':np.array(two_motifs_pos_scores).flatten(),
                            'cell line': np.tile(targets, two_motifs_pos_scores.shape[0])})
        df['motif 1'] = '{}_{}'.format(optimized_motifs[0], first_motif_pos)
        df.to_csv(distance_path, index=None)
    return df


def test_interaction(gi, optimized_motifs, targets, output_dir, filename):
    motifs_to_test = [ [(optimized_motifs[0][0], optimized_motifs[0][1])],
                            [(optimized_motifs[1][0], optimized_motifs[1][1])],
                            [(optimized_motifs[0][0], optimized_motifs[0][1]), (optimized_motifs[1][0], optimized_motifs[1][1])] ]
    interaction_test_dfs = []
    for motif_to_test in motifs_to_test:
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m,p in motif_to_test])
        diff = gi.embed_predict_quant_effect(motif_to_test).mean(axis=1)
        df = pd.DataFrame({
                            'mean difference':np.array(diff).flatten(),
                            'cell line': np.tile(targets, diff.shape[0])})
        df['motif'] = pattern_label
        interaction_test_dfs.append(df)
    pd.concat(interaction_test_dfs).to_csv(os.path.join(output_dir, filename))


def analyze_motif_pair(run_path, data_dir, motif_cluster, cell_lines, out_dir='GIA_results',
                       n_background=1000, motif1_positions=[1024]):
    """
    GIA addition experiment with one or pair of motifs. This optimizes any positions marked by '.' in the motifs, finds
    the optimal distance in case of pair of motifs and then tests for interaction by inserting them separately or together
    :param run_path: model path
    :param data_dir: test set dir
    :param motif_cluster:  iterable of motif strings
    :param cell_lines: iterable of strings of target names or integers of indices
    :param out_dir: output dir
    :param n_background: N sample background
    :param motif1_positions: positions where to put first motif in the interaction test
    :return: None
    """
    utils.make_dir(out_dir)  # make output dir
    testset, targets = evaluate.collect_whole_testset(data_dir=data_dir, coords=True)  # get test set
    C, X, Y = utils.convert_tfr_to_np(testset) # convert to np arrays for easy filtering
    model, _ = utils.read_model(run_path)  # load model

    run_name = os.path.basename(os.path.abspath(run_path))  # get identifier for the outputs
    gia_add_dir = utils.make_dir(os.path.join(out_dir, run_name))  # make a subdirectory for outputs

    # select background sequences to add the motif(s) to
    X_set = select_set('all_threshold', C, X, Y)


    gi = GlobalImportance(model, targets)
    gi.set_null_model(background_model, base_sequence=X_set, num_sample=n_background) # subsample background to given size
    for cell_line_name in cell_lines: # for each cell line of interest
        if isinstance(cell_line_name, int):
            cell_line_name = targets[cell_line_name]
        optimized_motifs = []
        for motif in motif_cluster: #
            print('Optimizing motif ' + motif)
            # make subdir for cell line and motif
            base_dir = utils.make_dir(os.path.join(gia_add_dir, '{}_{}'.format(cell_line_name, motif)))
            # subdir specific for a given background and number of samples
            output_dir = utils.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, n_background)))
            flanks_path = os.path.join(output_dir, 'flanks.csv')
            # get best motif by optimizing positions that are '.'
            optimized_motifs.append(record_flank_test(gi, motif, targets,
                                                      cell_line_name, flanks_path))
        if len(motif_cluster) == 2: # if two motifs are given
            print('Testing distance effect on motif interaction')
            # check for positional interaction by fixing one in the middle and sliding the other
            # subdirs for interaction results
            base_dir = utils.make_dir(
                os.path.join(gia_add_dir, '{}_{}_and_{}'.format(cell_line_name, motif_cluster[0], motif_cluster[1])))
            output_dir = utils.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, n_background)))
            # test_interaction(gi, optimized_motifs, targets, output_dir, 'optimal_position_interaction.csv')
            for first_motif_pos in motif1_positions:
                distance_path = os.path.join(output_dir, str(first_motif_pos) + '_distance.csv')
                # fix motif 1, shift motif 2 to find position that yields biggest importance score
                df = optimize_distance(gi, optimized_motifs, targets, distance_path, first_motif_pos)
                # get best position
                best_position = int(
                    df[df['cell line'] == cell_line_name].sort_values('mean difference').iloc[-1, 0].split('_')[1])
                motif_pos_pairs = [(optimized_motifs[0], first_motif_pos),
                                   (optimized_motifs[1], best_position)]
                # test for interaction using optimized flanks and distance
                test_interaction(gi, motif_pos_pairs, targets, output_dir,
                                 str(first_motif_pos) + '_best_distance_interaction.csv')
