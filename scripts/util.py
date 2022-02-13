import sys
import json
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from natsort import natsorted
import tensorflow as tf

def bin_resolution(y,bin_size):
    y_dim = y.shape
    y_bin = tf.math.reduce_mean(tf.reshape(y,(y_dim[0],int(y_dim[1]/bin_size),bin_size,y_dim[2])),axis = 2)
    return y_bin

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path

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



def make_dataset(data_dir, split_label, data_stats, batch_size=64, seed=None,
                shuffle=True, coords=False, drop_remainder=False):
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
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

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
