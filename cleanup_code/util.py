import sys
import json
import os
import time
import glob
from scipy.stats import pearsonr
import shutil
import hashlib
import sys
import pandas as pd
from sklearn.metrics import r2_score
import time
import numpy as np
import tensorflow as tf
from natsort import natsorted
import tensorflow as tf
from collections import OrderedDict
# import metrics
import scipy
import yaml

 ################################################################
 # functions for loading tfr files into tfr dataset
 ################################################################
def dna_one_hot(seq, seq_len=None, flatten=True, n_random=False):
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_random:
        seq_code = np.zeros((seq_len, 4), dtype="bool")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="float16")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_random:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
                else:
                    seq_code[i, :] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None, :]

        return seq_vec
    else:
        return seq_code


def hash_sequences_1hot(fasta_file, extend_len=None):
    # determine longest sequence
    if extend_len is not None:
        seq_len = extend_len
    else:
        seq_len = 0
        seq = ""
        for line in open(fasta_file):
            if line[0] == ">":
                if seq:
                    seq_len = max(seq_len, len(seq))

                header = line[1:].rstrip()
                seq = ""
            else:
                seq += line.rstrip()

        if seq:
            seq_len = max(seq_len, len(seq))

    # load and code sequences
    seq_vecs = OrderedDict()
    seq = ""
    for line in open(fasta_file):
        if line[0] == ">":
            if seq:
                seq_vecs[header] = dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ""
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = dna_one_hot(seq, seq_len)

    return seq_vecs

def bin_resolution(y,bin_size):
    y_dim = y.shape
    y_bin = tf.math.reduce_mean(tf.reshape(y,(y_dim[0],int(y_dim[1]/bin_size),bin_size,y_dim[2])),axis = 2)
    return y_bin

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path

def writ_list_to_file(string_list, file):
    with open(file, 'w') as filehandle:
        for line in string_list:
            filehandle.write(line)

def replace_all(text):
    dic = {'X': '23', 'Y': '24'}
    for i, j in dic.items():
        text = text.replace(i, j)
    return text.split('_')

def load_stats(data_dir):
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
      data_stats = json.load(data_stats_open)
  return data_stats

def batches_per_epoch(num_seqs, batch_size):
  return num_seqs // batch_size

def file_to_records(filename):
  return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def generate_parser(seq_length, target_length, num_targets, coords):
  def parse_proto(example_protos):
    """Parse TFRecord protobuf."""
    # TFRecord constants
    TFR_COORD = 'coordinate'
    TFR_INPUT = 'sequence'
    TFR_OUTPUT = 'target'

    # define features
    features = {
      TFR_COORD: tf.io.FixedLenFeature([], tf.string),
      TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
      TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string)
    }

    # parse example into features
    parsed_features = tf.io.parse_single_example(example_protos, features=features)

    # decode coords
    coordinate = parsed_features[TFR_COORD]

    # decode sequence
    # sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
    sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.float16)
    sequence = tf.reshape(sequence, [seq_length, 4])
    sequence = tf.cast(sequence, tf.float32)

    # decode targets
    targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
    targets = tf.reshape(targets, [target_length, num_targets])
    targets = tf.cast(targets, tf.float32)
    if coords:
        return coordinate, sequence, targets
    else:
        return sequence, targets

  return parse_proto



def make_dataset(data_dir, split_label, data_stats, batch_size=64, seed=None, shuffle=True, coords=False):
    seq_length = data_stats['seq_length']
    target_length = data_stats['target_length']
    num_targets = data_stats['num_targets']
    tfr_path = '%s/tfrecords/%s-*.tfr' % (data_dir, split_label)
    num_seqs = data_stats['%s_seqs' % split_label]

    tfr_files = natsorted(glob.glob(tfr_path))
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)

    # train
    # if split_label == 'train':
    if (split_label == 'train'):
      # repeat
      #dataset = dataset.repeat()

      # interleave files
      dataset = dataset.interleave(map_func=file_to_records,
        cycle_length=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # shuffle
      dataset = dataset.shuffle(buffer_size=32,
        reshuffle_each_iteration=True)

    # valid/test
    else:
      # flat mix files
      dataset = dataset.flat_map(file_to_records)

    dataset = dataset.map(generate_parser(seq_length, target_length, num_targets, coords))
    if shuffle:
        if seed:
            dataset = dataset.shuffle(32, seed=seed)
        else:
            dataset = dataset.shuffle(32)
    # dataset = dataset.batch(64)
    # batch
    dataset = dataset.batch(batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def make_dataset_binned(data_dir, split_label, data_stats, batch_size=64, seed=None, shuffle=True, coords=False, bin_size=1):
    seq_length = data_stats['seq_length']
    target_length = data_stats['target_length']
    num_targets = data_stats['num_targets']
    tfr_path = '%s/tfrecords/%s-*.tfr' % (data_dir, split_label)
    num_seqs = data_stats['%s_seqs' % split_label]

    tfr_files = natsorted(glob.glob(tfr_path))
    dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)

    dataset = dataset.flat_map(file_to_records)

    dataset = dataset.map(generate_parser_with_binning(seq_length, target_length, num_targets, coords, bin_size=128))
    if shuffle:
        if seed:
            dataset = dataset.shuffle(32, seed=seed)
        else:
            dataset = dataset.shuffle(32)
    dataset = dataset.batch(64)
    # batch
    dataset = dataset.batch(batch_size)

    # prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def tfr_to_np(data, choose, array_shape):
    if choose=='x':
        data_part = data.map(lambda z,x,y: x)
    elif choose=='y':
        data_part = data.map(lambda z,x,y: y)
    data_np = np.zeros(array_shape)
    # load data to a numpy array
    iter_data = iter(data_part)
    j=0
    for i in iter_data:
        n_seqs = i.shape[0]
        data_np[j:j+n_seqs,:,:] = i
        j+=n_seqs
    return data_np

def window_shift(X,Y,window_size,shift_num,both_seq = False):
    if len(X.shape) == 2:
        X = np.expand_dims(X,axis = 0)

    chop_size = X.shape[1]
    input_seq_num = X.shape[0]
    output_num = shift_num*input_seq_num
    #Shift X around
    ori_X = np.repeat(X,shift_num,axis=0)
    shift_idx = (np.arange(window_size) +
                np.random.randint(low = 0,high = chop_size-window_size,
                                  size = output_num)[:,np.newaxis])
    col_idx = shift_idx.reshape(window_size *output_num)
    row_idx = np.repeat(range(0,output_num),window_size)
    f_index = np.vstack((row_idx,col_idx)).T.reshape(output_num,window_size,2)
    shift_x = tf.gather_nd(ori_X,f_index)

    #shift Y accordingly
    if both_seq == True and len(Y.shape)==2:
        Y = np.expand_dims(Y,axis = 0)
    ori_Y = np.repeat(Y,shift_num,axis=0)
    shift_y = tf.gather_nd(ori_Y,f_index)

    shift_idx = shift_idx[:,0]
    center_idx = int(0.5*(chop_size-window_size))
    relative_shift_idx =shift_idx - center_idx

    return np.array(shift_x),np.array(shift_y),relative_shift_idx

def evaluate_shift(X,Y,model,window_size,shift_num):
    shift_x,shift_y,shift_idx = window_shift(X,Y,window_size,shift_num)
    pred_y = model.predict(shift_x)
    if pred_y.shape[1] < shift_y.shape[1]:
        bin_size =  int(shift_y.shape[1] / pred_y.shape[1])
        pred_y = np.repeat(pred_y,bin_size,axis = 1)
        shift_y = bin_resolution(shift_y,bin_size)
        shift_y = np.repeat(shift_y,bin_size,axis=1)
    p_r_list = []
    for i,pred in enumerate(pred_y):
        task_pr = []
        for task in range(pred.shape[1]):
            p_r = scipy.stats.pearsonr(shift_y[i,task],pred[task])[0]
            task_pr.append(p_r)
        p_r_list.append(task_pr)
    return np.array(p_r_list),np.array(shift_idx)


class SeabornFig2Grid():
    '''Class for plotting multiple sns jointplots'''

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or             isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def convert_tfr_to_np(testset, number_data_types=2):
    all_data = [[] for i in range(number_data_types)]
    for i, (data) in enumerate(testset):
        for j, data_type in enumerate(data):
            all_data[j].append(data_type)
    return [np.concatenate(d) for d in all_data]


def batch_np(whole_dataset, batch_size):
    for i in range(0, whole_dataset.shape[0], batch_size):
        yield whole_dataset[i:i+batch_size]

def onehot_to_str(onehot):
    full_str = []
    for one_onehot in onehot:
        assert one_onehot.shape == (4,)
        full_str.append(list('ACGT')[np.argwhere(one_onehot)[0][0]])
    return ''.join(full_str)
