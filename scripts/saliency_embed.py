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
