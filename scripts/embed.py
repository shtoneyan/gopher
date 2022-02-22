import tfr_evaluate
import subprocess
import numpy as np
import pandas as pd
import utils
import umap.umap_ as umap


def select(embeddings, lower_lim_1=None,
           upper_lim_1=None, lower_lim_2=None,
           upper_lim_2=None, idr=''):
    '''
    This funciton selects embedding points of a UMAP for downstream tasks
    :param embeddings: UMAP 2D embeddings in pandas dataframe
    :param lower_lim_1: X axis lower lim
    :param upper_lim_1: X axis upper lim
    :param lower_lim_2: Y axis lower lim
    :param upper_lim_2: Y axis upper lim
    :param idr: if 'y' filter only IDR peaks if 'n' only non-IDR peaks (if not set to anything take all)
    :return: mask filter
    '''
    mask = np.zeros((embeddings['UMAP 1'].shape[0])) + 1
    if lower_lim_1:
        mask *= (embeddings['UMAP 1'] > lower_lim_1).values
    if upper_lim_1:
        mask *= (embeddings['UMAP 1'] < upper_lim_1).values
    if lower_lim_2:
        mask *= (embeddings['UMAP 2'] > lower_lim_2).values
    if upper_lim_2:
        mask *= (embeddings['UMAP 2'] < upper_lim_2).values
    if idr == 'y':
        print('Choosing only IDR')
        mask *= (embeddings['IDR'] == True).values
    if idr == 'n':
        print('Choosing only non IDR')
        mask *= (embeddings['IDR'] != True).values
    return mask.astype(bool)


def get_cell_line_overlaps(file_prefix='cell_line_13',
                           bedfile1='/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/sequences.bed',
                           bedfile2='/home/shush/profile/QuantPred/datasets/cell_line_specific_test_sets/cell_line_13/complete/peak_centered/i_2048_w_1.bed',
                           fraction_overlap=0.5):
    """
    This function filters overlapping bed ranges and returns start coordinates of points that have idr overlaps. Useful for "annotating" whole chromosome chunks as idr or non-idr
    :param file_prefix: output csv file prefix
    :param bedfile1: first bedfile (the ranges of which will be annotated)
    :param bedfile2: second bedfile that contains the idr regions
    :param fraction_overlap: minimum fraction of overlap needed to call a sequence idr
    :return: vector of starting positions of idr sequences in the test set
    """
    cmd = 'bedtools intersect -f {} -wa -a {} -b {} | uniq > {}_IDR.bed'.format(fraction_overlap, bedfile1, bedfile2, file_prefix)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    _ = process.communicate()
    df = pd.read_csv('{}_IDR.bed'.format(file_prefix), sep='\t', header=None)
    idr_starts = df.iloc[:, 1].values
    return idr_starts


def threshold_cell_line_np(np_C, np_X, np_Y, cell_line, more_than, less_than=None):
    """
    Threshold based on cell line specific coverage values.
    :param np_C: np array of coordinates
    :param np_X: np array of onehot sequences
    :param np_Y: np array of target coverage values
    :param cell_line: cell line number or index
    :param more_than: lower limit
    :param less_than: upper limit
    :return: filtered coordinates, onehot sequences and targets
    """
    m1 = np_Y[:, :, cell_line].max(axis=1) > more_than
    if less_than:
        m2 = np_Y[:, :, cell_line].max(axis=1) < less_than
        threshold_mask = (m1 & m2)
    else:
        threshold_mask = m1
    thresholded_X = np_X[threshold_mask]
    thresholded_C = np_C[threshold_mask]
    thresholded_Y = np_Y[threshold_mask, :, cell_line]
    return (thresholded_C, thresholded_X, thresholded_Y)


def label_idr_peaks(C, cell_line):
    """
    Function to classify each coordinate as idr or non-idr
    :param C: iterable of coordinates in the format saved in the tfr datasets
    :param cell_line: cell line to select the corresponding IDR peaks
    :return: list of boolean values indicating if peak is present at that coordinate
    """
    idr_class = []
    idr_starts = get_cell_line_overlaps(cell_line)
    idr_class.append([True if int(str(c).strip('\'b').split('_')[1]) in idr_starts else False for c in C])
    idr_class = [item for sublist in idr_class for item in sublist]
    return idr_class


def predict_np(X, model, batch_size=32, reshape_to_2D=False):
    """
    Function to get intermediate representations or predictions from a model
    :param X: onehot sequences
    :param model: trained model loaded into memory
    :param batch_size: batch size
    :param reshape_to_2D: bool, if true reshape to 2D for UMAP
    :return:
    """
    model_output = []
    for x_batch in util.batch_np(X, batch_size):
        model_output.append(model(x_batch).numpy())
    model_output = np.squeeze(np.concatenate(model_output))
    if reshape_to_2D:
        assert len(model_output.shape) == 3, 'Wrong dimension for reshape'
        d1, d2, d3 = model_output.shape
        model_output = model_output.reshape(d1, d2 * d3)
    return model_output


def get_embeddings(input_features):
    """
    This function puts embeddings as a pandas dataframe
    :param input_features: intermediate representations
    :return: pandas dataframe of embeddings
    """
    reducer = umap.UMAP(random_state=28)
    embedding = reducer.fit_transform(input_features)
    df = pd.DataFrame({'UMAP 1': embedding[:, 1], 'UMAP 2': embedding[:, 0]})
    return df


