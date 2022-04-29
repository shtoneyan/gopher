import h5py
import itertools
import numpy as np
import os
import pandas as pd
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
        select sequences to remove/occlude a given motif from by randomizing nucleotides
        :param subset: a set of onehot sequences in np array
        :param num_sample: number to limit the sequences to
        :param seed: random sample seed
        :return:
        """
        if num_sample:
            if seed:
                np.random.seed(seed)
            rand_idx = np.random.choice(subset.shape[0], num_sample, replace=False).flatten()
            self.seqs_to_remove_motif = subset[rand_idx]
        else:
            self.seqs_to_remove_motif = subset

    def occlude_all_motif_instances(self, subset, tandem_motifs_to_remove,
                                    num_sample=None,
                                    seed=42, batch_size=32):
        """

        :param subset: subset of sequences to occlude the motif in
        :param tandem_motifs_to_remove: list of motifs to remove in tandem/together
        :param num_sample: sample size
        :param seed: random seed for sampling
        :param batch_size: batch size for making predictions
        :return: None
        """
        self.set_seqs_for_removing(subset, num_sample, seed)
        print('tandem_motifs_to_remove', tandem_motifs_to_remove)
        motif_key = ', '.join(tandem_motifs_to_remove)
        print(motif_key)
        self.seqs_with[motif_key], self.seqs_removed[motif_key], self.n_instances, self.seq_idx[
            motif_key] = randomize_multiple_seqs(self.seqs_to_remove_motif,
                                                 tandem_motifs_to_remove)
        if len(self.seqs_with[motif_key]) > 0:
            self.seqs_with[motif_key], self.seqs_removed[motif_key] = [np.array(n) for n in [self.seqs_with[motif_key],
                                                                                             self.seqs_removed[
                                                                                                 motif_key]]]
            df = self.get_predictions(motif_key, batch_size)
        else:
            print('WARNING: no seqs with motifs found')
            df = pd.DataFrame(
                {'mean coverage': [None], 'sequence': [None], 'N instances': [None], 'motif pattern': [motif_key]})
        self.summary_remove_motifs.append(df)

    def get_predictions(self, motif_key, batch_size):
        """
        This function gets the predictions for original sequences and same sequences with occluded motifs
        :param motif_key: label of the motif
        :param batch_size: batch size for predictions
        :return: dataframe summary of the predictions and metadata
        """
        # predicted coverage for original sequences
        ori_preds = utils.predict_np((self.seqs_with[motif_key]),
                                     self.model, batch_size=batch_size,
                                     reshape_to_2D=False)
        # predicted coverage for sequences with occluded motifs
        del_preds = get_avg_preds(self.seqs_removed[motif_key],
                                  self.model)
        # if binary model
        if ori_preds.ndim == 2 and del_preds.ndim == 2:
            ori_preds = np.expand_dims(ori_preds, axis=1)
            del_preds = np.expand_dims(del_preds, axis=1)
        max_ori_pc3 = np.mean(make_3D(ori_preds), axis=1) # get mean prediction per original sequence
        max_pred_pc3 = np.mean(make_3D(del_preds), axis=1) # get mean prediction per occluded sequence
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

    def set_null_model(self, null_model, base_sequence, num_sample=1000, seed=None):
        """
        use model-based approach to set the null sequences
        :param null_model: dinuc or none - approach to select background
        :param base_sequence: set of onehot sequences
        :param num_sample: number of sequences
        :param seed: optional seed for random selection
        :return:
        """
        self.x_null = generate_null_sequence_set(null_model, base_sequence, num_sample, seed)
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
        """
        embed patterns in null sequences
        :param patterns: iterable of tuples of str pattern and position for where to insert it in the sequence
        :return: onehot sequence with motif embedding
        """
        if not isinstance(patterns, list):
            patterns = [patterns]

        x_index = np.copy(self.x_null_index)  # argmax version of onehot background sequences
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
        """
        embed pattern in null sequences and get their predictions
        :param patterns: patterns/motifs to embed
        :return: difference between predictions in case of embedded sequence and original background
        """
        one_hot = self.embed_patterns(patterns)  # create sequences with embedded motifs
        # id for the motif indicating submotifs and insertion sites
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m, p in patterns])
        self.embedded_predictions[pattern_label] = self.model.predict(one_hot)  # save predictions in a dict
        assert self.embedded_predictions[pattern_label].shape == self.null_profiles.shape
        if self.embedded_predictions[pattern_label].ndim == 2:  # if from binary model expand to match quantitative
            return np.expand_dims(self.embedded_predictions[pattern_label] - self.null_profiles, axis=1)
        else:
            return self.embedded_predictions[pattern_label] - self.null_profiles  # return delta predictions

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
def generate_null_sequence_set(null_model, base_sequence, num_sample=1000, seed=None):
    """
    make a subset for background based on null model type
    :param null_model: startegy for generating the background sequences
    :param base_sequence: sequences to use for generating backgrounds
    :param num_sample: sample size
    :param seed: seed for random choice for null model none
    :return: None
    """
    if null_model == 'random':    return generate_shuffled_set(base_sequence, num_sample)  # shuffle
    if null_model == 'profile':   return generate_profile_set(base_sequence, num_sample)  # match nucl profile
    if null_model == 'dinuc':     return generate_dinucleotide_shuffled_set(base_sequence, num_sample)  # dinuc shuffle
    if null_model == 'none':  # no shuffle, just subset
        if seed:
            np.random.seed(seed)
        idx = np.random.choice(base_sequence.shape[0], num_sample)
        return base_sequence[idx]
    else:
        print ('null_model name not recognized.')


def generate_profile_set(base_sequence, num_sample):
    """
    create a subset of sequences as background by matching nucleotide profiles
    :param base_sequence: sequences to use for matching
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
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
    """
    Funciton for creating a shuffled set of sequences based on an input set
    :param base_sequence: sequences to shuffle
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle nucleotides
    [np.random.shuffle(x) for x in x_null]
    return x_null


def generate_dinucleotide_shuffled_set(base_sequence, num_sample):
    """
    Function for dinuc shuffling provided sequences
    :param base_sequence: set of sequences
    :param num_sample: sample size
    :return: background set of onehot sequences
    """
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle dinucleotides
    for j, seq in enumerate(x_null):
        x_null[j] = dinuc_shuffle(seq)
    return x_null


# -------------------------------------------------------------------------------------
# util functions
# -------------------------------------------------------------------------------------
def select_set(testset_type, C, X, Y, cell_line=None):
    """
    This function selects sequences for constructing background sequences
    :param testset_type:
    :param C: coordinates
    :param X: sequences
    :param Y: target coverages
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
    """
    Function to reshape array if not 3D
    :param array: np array
    :return: either same array or reshaped into 3D
    """
    if len(array.shape) == 2:
        return np.expand_dims(array, axis=0)
    elif len(array.shape) == 3:
        return array
    else:
        print('bad array')
        exit()



# -------------------------------------------------------------------------------------
# functions to find a motif in a sequence
# -------------------------------------------------------------------------------------



def select_indices(motif_pattern, str_seq):
    '''
    select indices according to filtering criteria
    :param motif_pattern: string to search for
    :param str_seq: sequence string
    :return: indices where the substring is found
    '''
    iter = re.finditer(motif_pattern, str_seq)
    indices = [m.start(0) for m in iter]
    return indices


def find_multiple_motifs(motif_pattern_list, str_seq):
    '''
    find indices of multiple motifs in a single sequence
    :param motif_pattern_list: list of string motif patterns
    :param str_seq: string form of the sequence
    :return:
    '''
    motifs_and_indices = {}
    for motif_pattern in motif_pattern_list:
        chosen_ind = select_indices(motif_pattern, str_seq)
        motifs_and_indices[motif_pattern] = chosen_ind
    return motifs_and_indices


# -------------------------------------------------------------------------------------
# functions to remove or randomize a motif
# -------------------------------------------------------------------------------------

def randomize_motif_dict_in_seq(motifs_and_indices, selected_seq, n_occlusions=25):
    """

    :param motifs_and_indices: motifs and positions where they occur in a sequence
    :param selected_seq: a single sequence where the motif will be randomized
    :param n_occlusions: number of times to randomize the motif nucleotides
    :return: an array of sequences with randomly occluded motifs
    """
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


def randomize_multiple_seqs(onehot_seqs, tandem_motifs_to_remove):
    """

    :param onehot_seqs: iterable of onehot sequences
    :param tandem_motifs_to_remove: list of motifs
    :return: original sequences, sequences with the motif randomized, the number of times the motif(s) occur per sequence
    and indices where the motif is randomized
    """
    seqs_with_motif = []
    seqs_removed_motifs = []
    n_instances_per_seq = []
    incl_idx = []
    for o, onehot_seq in tqdm(enumerate(onehot_seqs)):
        str_seq = ''.join(utils.onehot_to_str(onehot_seq))
        motifs_and_indices = find_multiple_motifs(tandem_motifs_to_remove, str_seq)
        all_motifs_present = np.array([len(v) > 0 for k, v in motifs_and_indices.items()]).all()
        if all_motifs_present:
            seqs_with_motif.append(onehot_seq.copy())
            seqs_removed_motifs.append(randomize_motif_dict_in_seq(motifs_and_indices,
                                                                   onehot_seq))
            n_instances_per_seq.append([str(len(v)) for k, v in motifs_and_indices.items()])
            incl_idx.append(o)
    n_instances_per_seq = [', '.join(n) for n in n_instances_per_seq]
    return seqs_with_motif, seqs_removed_motifs, n_instances_per_seq, incl_idx


def get_avg_preds(seqs_removed, model):
    """

    :param seqs_removed: sequences with randomized motifs
    :param model: model object
    :return: predictions per sequence averaged across all randomizations
    """
    N, B, L, C = seqs_removed.shape # extra channel for per sequence random occlusions
    removed_preds = utils.predict_np((seqs_removed.reshape(N * B, L, C)), model,
                                     batch_size=32, reshape_to_2D=False) # all predictions from occluded sequences
    # add axis if binary model, reshape back into per sequence and get mean prediction per sequence
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
    """
    Function to measure the global importance of each flank
    :param gi: GlobalImportance class instance
    :param all_flanks: iterable of all motif verisons
    :param targets: all targets
    :param position: sequence position to put the motif in
    :param output_path: path for csv to save results
    :return: dataframe with flanks and global importance scores per target
    """
    all_scores = []
    for motif in tqdm(all_flanks):
        diff_scores = gi.embed_predict_quant_effect([(motif, position)])
        all_scores_per_motif = (diff_scores).mean(axis=0).mean(axis=0)  # compute mean to get global importance score
        all_scores.append(all_scores_per_motif)
    df = pd.DataFrame({'motif': np.repeat(all_flanks, len(targets)),
                       'mean difference': np.array(all_scores).flatten(),
                       'cell line': np.tile(targets, len(all_flanks))})
    df.to_csv(output_path, index=None)
    return df


def generate_flanks(motif_pattern):
    """
    Function to create a set of all possible kmers in the given motif flanks or gaps
    :param motif_pattern: string of motif to create flanks for
    :return: all possible complete motifs with flanks or gaps
    """
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


def record_flank_test(gi, motif, targets, cell_line_name, flanks_path):
    """
    Function for generating and testing each flank or motif variant global importance
    :param gi: GlobalImportance class instance
    :param motif: motif string (gaps indicated with dots '.')
    :param targets: all targets iterable
    :param cell_line_name: cell line or target to use for selecting motif with biggest global importance
    :param flanks_path: path to the csv file where the results are saved for each flank
    :return: flank with biggest global importance for the given cell line
    """
    # select the best flanks based on where the dots are in the pattern
    if '.' in motif:
        if os.path.isfile(flanks_path):
            flank_scores = pd.read_csv(flanks_path)
        else:
            all_motifs = generate_flanks(motif)
            print('Testing flanks')
            flank_scores = test_flanks(gi, all_motifs, targets,
                                       output_path=flanks_path)
        best_flank = flank_scores[flank_scores['cell line'] == cell_line_name].sort_values('mean difference').iloc[
            -1, 0]
        print(best_flank)
    else:
        best_flank = motif
    return best_flank


def optimize_distance(gi, optimized_motifs, targets, distance_path, first_motif_pos):
    """
    Function to get the optimal distance between two motifs for a given cell line of interest
    :param gi: GlobalImportance class instance
    :param optimized_motifs: motif with no '.' in the string
    :param targets: target labels
    :param distance_path: path where to save csv
    :param first_motif_pos: position where to insert the first motif as the second's position is changed
    :return: dataframe with distance information
    """
    if os.path.isfile(distance_path):
        df = pd.read_csv(distance_path)
    else:
        two_motifs_pos_scores = []
        positions = list(range(0, 2048 - len(optimized_motifs[1]), 2))
        for position in tqdm(positions):
            diff_scores = gi.embed_predict_quant_effect([(optimized_motifs[0], first_motif_pos),
                                                         (optimized_motifs[1],
                                                          position)])
            two_motifs_pos_scores.append(diff_scores.mean(axis=0).mean(axis=0))
        two_motifs_pos_scores = np.array(two_motifs_pos_scores)
        motif_2_label = np.array(['{}_{}'.format(optimized_motifs[1], pos) for pos in positions])
        df = pd.DataFrame({
            'motif 2': np.repeat(motif_2_label, len(targets)),
            'mean difference': np.array(two_motifs_pos_scores).flatten(),
            'cell line': np.tile(targets, two_motifs_pos_scores.shape[0])})
        df['motif 1'] = '{}_{}'.format(optimized_motifs[0], first_motif_pos)
        df.to_csv(distance_path, index=None)
    return df


def test_interaction(gi, optimized_motifs, targets, output_dir, filename):
    """
    Function to test interaction between 2 motifs by computing global importance of each inserted individually or
    in combination.
    :param gi: GlobalImportance class instance
    :param optimized_motifs: 2 motif strings and positions where to insert them
    :param targets: target labels
    :param output_dir: dir path to save results
    :param filename: csv filename
    :return: None
    """
    # make single or combined motif clusters to embed
    motifs_to_test = [[(optimized_motifs[0][0], optimized_motifs[0][1])],
                      [(optimized_motifs[1][0], optimized_motifs[1][1])],
                      [(optimized_motifs[0][0], optimized_motifs[0][1]),
                       (optimized_motifs[1][0], optimized_motifs[1][1])]]
    interaction_test_dfs = []
    for motif_to_test in motifs_to_test:  # for each in [motif1, motif2, motif1_and_motif2]
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m, p in motif_to_test])
        diff = gi.embed_predict_quant_effect(motif_to_test).mean(axis=1)  # get mean global importance
        df = pd.DataFrame({
            'mean difference': np.array(diff).flatten(),
            'cell line': np.tile(targets, diff.shape[0])})
        df['motif'] = pattern_label
        interaction_test_dfs.append(df)  # save in a dataframe
    pd.concat(interaction_test_dfs).to_csv(os.path.join(output_dir, filename))  # combine all and write to csv


def gia_add_motifs(run_path, data_dir, motif_cluster, cell_lines, out_dir='GIA_results',
                       n_background=1000, motif1_positions=[1024], background_model='dinuc'):
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
    :param background_model: method for generating background sequences
    :return: None
    """
    utils.make_dir(out_dir)  # make output dir
    testset, targets = utils.collect_whole_testset(data_dir=data_dir, coords=True)  # get test set
    C, X, Y = utils.convert_tfr_to_np(testset)  # convert to np arrays for easy filtering
    model, _ = utils.read_model(run_path)  # load model
    run_name_found = [r for r in os.path.abspath(run_path).split('/') if 'run-' in r]  # get identifier for the outputs
    if run_name_found:
        run_name = run_name_found[0]
    else:
        run_name = 'results_' + str(hash(run_path))
    print('Saving results in subfolder '+run_name)
    gia_add_dir = utils.make_dir(os.path.join(out_dir, run_name))  # make a subdirectory for outputs

    # select background sequences to add the motif(s) to
    X_set = select_set('all_threshold', C, X, Y)

    gi = GlobalImportance(model, targets)
    # subsample background to given size
    gi.set_null_model(background_model, base_sequence=X_set, num_sample=n_background)
    for cell_line_name in cell_lines:  # for each cell line of interest
        if isinstance(cell_line_name, int):
            cell_line_name = targets[cell_line_name]
        optimized_motifs = []
        for motif in motif_cluster:
            print('Optimizing motif ' + motif)
            # make subdir for cell line and motif
            base_dir = utils.make_dir(os.path.join(gia_add_dir, '{}_{}'.format(cell_line_name, motif)))
            # subdir specific for a given background and number of samples
            output_dir = utils.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, n_background)))
            flanks_path = os.path.join(output_dir, 'flanks.csv')
            # get best motif by optimizing positions that are '.'
            optimized_motifs.append(record_flank_test(gi, motif, targets,
                                                      cell_line_name, flanks_path))
        if len(motif_cluster) == 2:  # if two motifs are given
            print('Testing distance effect on motif interaction')
            # check for positional interaction by fixing one in the middle and sliding the other
            # subdirs for interaction results
            base_dir = utils.make_dir(
                os.path.join(gia_add_dir, '{}_{}_and_{}'.format(cell_line_name, motif_cluster[0], motif_cluster[1])))
            output_dir = utils.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, n_background)))
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


def gia_occlude_motifs(run_path, data_dir, motif_cluster, X_subset_type='all_threshold', out_dir='GIA_occlude_results',
                       n_background=1000):
    """
    Function for testing effect of randomizing or occluding a set of motifs in a sequence individually or together
    :param run_path: model run path
    :param data_dir: directory with the test data
    :param motif_cluster: list of motifs to test
    :param X_subset_type: method for subsetting test set, all_threshold means it filters the dataset using all target coverage values
    :param out_dir: output directory
    :param n_background: number of sequences to sample and search the motifs in
    :return: None
    """
    utils.make_dir(out_dir)  # make output dir
    testset, targets = utils.collect_whole_testset(data_dir=data_dir, coords=True)  # get test set
    C, X, Y = utils.convert_tfr_to_np(testset)  # convert to np arrays for easy filtering
    model, _ = utils.read_model(run_path)  # load model
    run_name = os.path.basename(os.path.abspath(run_path))  # get identifier for the outputs
    gia_occ_dir = utils.make_dir(os.path.join(out_dir, run_name))
    output_dir = utils.make_dir(os.path.join(gia_occ_dir, '{}_N{}'.format(X_subset_type, n_background)))
    X_set = select_set(X_subset_type, C, X, Y)
    gi = GlobalImportance(model, targets)
    if len(motif_cluster) > 1:
        combo_list = [motif_cluster] + [[m] for m in motif_cluster]
    else:
        combo_list = motif_cluster
    for each_element in combo_list:
        print(each_element)
        gi.occlude_all_motif_instances(X_set, each_element, num_sample=n_background)
    df = pd.concat(gi.summary_remove_motifs) # collect all dataframes with results
    file_prefix = '&'.join(motif_cluster)
    df.to_csv(os.path.join(output_dir, file_prefix+'.csv'), index=None)
