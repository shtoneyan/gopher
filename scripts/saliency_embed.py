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



def plot_saliency(saliency_map):
    fig, axs = plt.subplots(saliency_map.shape[0], 1, figsize=(20, 2 * saliency_map.shape[0]))
    for n, w in enumerate(saliency_map):
        if saliency_map.shape[0] == 1:
            ax = axs
        else:
            ax = axs[n]
        # plot saliency map representation
        saliency_df = pd.DataFrame(w, columns=['A', 'C', 'G', 'T'])
        logomaker.Logo(saliency_df, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
    return plt



def select_top_pred(pred, num_task, top_num):
    task_top_list = []
    for i in range(0, num_task):
        task_profile = pred[:, :, i]
        task_mean = np.squeeze(np.mean(task_profile, axis=1))
        task_index = task_mean.argsort()[-top_num:]
        task_top_list.append(task_index)
    task_top_list = np.array(task_top_list)
    return task_top_list


def complete_saliency(X, model, class_index, func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
    # if not tf.is_tensor(X):
    #   X = tf.Variable(X)

    X = tf.cast(X, dtype='float32')

    with tf.GradientTape() as tape:
        tape.watch(X)
        if class_index is not None:
            outputs = func(model(X)[:, :, class_index])
        else:
            raise ValueError('class index must be provided')
    return tape.gradient(outputs, X)


def peak_saliency_map(X, model, class_index, window_size, func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        pred = model(X)

        peak_index = tf.math.argmax(pred[:, :, class_index], axis=1)
        batch_indices = []

        if int(window_size) > 50:
            bin_num = 1
        elif int(window_size) == 32:
            bin_num = 3
        else:
            bin_num = 50

        for i in range(0, X.shape[0]):
            column_indices = tf.range(peak_index[i] - int(bin_num / 2), peak_index[i] + math.ceil(bin_num / 2),
                                      dtype='int32')
            row_indices = tf.keras.backend.repeat_elements(tf.constant([i]), bin_num, axis=0)
            full_indices = tf.stack([row_indices, column_indices], axis=1)
            batch_indices.append([full_indices])
            outputs = func(tf.gather_nd(pred[:, :, class_index], batch_indices), axis=2)

        return tape.gradient(outputs, X)


class Explainer():
    """wrapper class for attribution maps"""

    def __init__(self, model, class_index=None, func=tf.math.reduce_mean, binary=False):
        self.model = model
        self.class_index = class_index
        self.func = func
        self.binary = binary

    def saliency_maps(self, X, batch_size=128):
        return function_batch(X, saliency_map, batch_size, model=self.model,
                              class_index=self.class_index, func=self.func,
                              binary=self.binary)


@tf.function
def saliency_map(X, model, class_index=None, func=tf.math.reduce_mean, binary=False):
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        if binary:
            outputs = model(X)[:, class_index]
        else:
            outputs = tf.math.reduce_mean(model(X)[:, :, class_index], axis=1)
    return tape.gradient(outputs, X)


def plot_mean_coverages(data_and_labels, ax):
    for i, (data, label, p) in enumerate(data_and_labels):
        if 'non' in label:
            marker_style = '--'
        else:
            marker_style = '-'
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
        ax.fill_between(x,cis[0], cis[1], alpha=0.08, color=p)
        ax.plot(x,  est,p,label=label)