import tfr_evaluate
import subprocess
import numpy as np
import pandas as pd
import utils
import umap.umap_ as umap
import tensorflow as tf
from tensorflow import keras


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

def tomtom_upstream(model_path,filter_layer,seq,output_path,threshold = 0.5, pad = 3):
    model = modelzoo.load_model(model_path,compile = True)
    max_filter,counter = filter_max_align_batch(seq,model,layer = filter_layer)
    clip_filter = clip_filters(max_filter,threshold = threshold, pad = pad)
    meme_generate(clip_filter,output_file = ooutput_path+'.txt')


def meme_generate(W, output_file='meme.txt', prefix='filter'):
  """generate a meme file for a set of filters, W ∈ (N,L,A)"""

  # background frequency
  nt_freqs = [1./4 for i in range(4)]

  # open file for writing
  f = open(output_file, 'w')

  # print intro material
  f.write('MEME version 4\n')
  f.write('\n')
  f.write('ALPHABET= ACGT\n')
  f.write('\n')
  f.write('Background letter frequencies:\n')
  f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
  f.write('\n')

  for j, pwm in enumerate(W):
    L, A = pwm.shape
    f.write('MOTIF %s%d \n' % (prefix, j))
    f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
    for i in range(L):
      f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
    f.write('\n')

  f.close()

def clip_filters(W, threshold=0.5, pad=3):
  """clip uninformative parts of conv filters"""
  W_clipped = []
  for w in W:
    L,A = w.shape
    entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
    index = np.where(entropy > threshold)[0]
    if index.any():
      start = np.maximum(np.min(index)-pad, 0)
      end = np.minimum(np.max(index)+pad+1, L)
      W_clipped.append(w[start:end,:])
    else:
      W_clipped.append(w)

  return W_clipped

def filter_max_align_batch(X, model, layer=3, window=24, threshold=0.5, batch_size=1024, max_align=1e4, verbose=1):
  """get alignment of filter activations for visualization"""
  if verbose:
    print("Calculating filter PPM based on activation-based alignments")
  N,L,A = X.element_spec.shape
  num_filters = model.layers[layer].output.shape[2]

  # Set the left and right window sizes
  window_left = int(window/2)
  window_right = window - window_left

  # get feature maps of 1st convolutional layer after activation
  intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

  #dataset = tf.data.Dataset.from_tensor_slices(X)
  #batches = X.batch(batch_size)
  batches = X
  # loop over batches to capture MAX activation
  if verbose:
    print('  Calculating MAX activation')
  MAX = np.zeros(num_filters)
  for x in batches:

    # get feature map for mini-batch
    fmap = intermediate.predict(x)

    # loop over each filter to find "active" positions
    for f in range(num_filters):
      MAX[f] = np.maximum(MAX[f], np.max(fmap[:,:,f]))


  # loop over each filter to find "active" positions

  W = []
  counts = []
  for f in range(num_filters):
    if verbose:
      print("    processing %d out of %d filters"%(f+1, num_filters))
    status = 0

    # compile sub-model to get feature map
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output[:,:,f])

    # loop over each batch
    #dataset = tf.data.Dataset.from_tensor_slices(X)
    seq_align_sum = np.zeros((window, A)) # running sum
    counter = 0                            # counts the number of sequences in alignment
    status = 1                            # monitors whether depth of alignment has reached max_align
    for x in X:
      if status:

        # get feature map for a batch sequences
        fmaps = intermediate.predict(x)

        # Find regions above threshold
        for data_index, fmap in enumerate(fmaps):
          if status:
            pos_index = np.where(fmap > MAX[f] * threshold)[0]

            # Make a sequence alignment centered about each activation (above threshold)
            for i in range(len(pos_index)):
              if status:
                # Determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # Check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                  seq_align_sum += x[data_index,start_window:end_window,:].numpy()
                  counter += 1
                  if counter > max_align:
                    status = 0
                else:
                  break
          else:
            break
      else:
        if verbose:
          print("      alignment has reached max depth for all filters")
        break

    # calculate position probability matrix of filter
    if verbose:
      print("      %d sub-sequences above threshold"%(counter))
    if counter > 0:
      W.append(seq_align_sum/counter)
    else:
      W.append(np.ones((window,A))/A)
    counts.append(counter)
  return np.array(W), np.array(counts)


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
