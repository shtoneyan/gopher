#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import os
import sys
import uuid
import h5py
import numpy as np
import pdb
import pysam
import json

from basenji_data import ModelSeq
from dna_io import dna_1hot, dna_1hot_index

import tensorflow as tf

"""
basenji_data_write.py

Write TF Records for batches of model sequences.

Notes:
-I think target_start and target_end are remnants of my previous data2 pipeline.
 If I see this again beyond 8/2020, remove it.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_cov_dir> <tfr_file> <fold_set>'
  parser = OptionParser(usage)
  parser.add_option('--threshold', dest='threshold',
      default=0, type='float',
      help='Set a minimum threshold for activity.')
  parser.add_option('--test_threshold', dest='test_threshold',
      type='float',
      help='Set a minimum threshold for activity for test set.')
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('--te', dest='target_extend',
      default=None, type='int', help='Extend targets vector [Default: %default]')
  parser.add_option('--ts', dest='target_start',
      default=0, type='int', help='Write targets into vector starting at index [Default: %default')
  parser.add_option('-u', dest='umap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--umap_clip', dest='umap_clip',
      default=1, type='float',
      help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
  parser.add_option('--umap_tfr', dest='umap_tfr',
      default=False, action='store_true',
      help='Save umap array into TFRecords [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='data_out',
      help='Output directory [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 5:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_cov_dir = args[2]
    tfr_file = args[3]
    fold_set = args[4]

  if fold_set == 'test':
      options.threshold = options.test_threshold

  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # determine sequence coverage files

  seqs_cov_files = []
  ti = 0
  seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)
  while os.path.isfile(seqs_cov_file):
    seqs_cov_files.append(seqs_cov_file)
    ti += 1
    seqs_cov_file = '%s/%d.h5' % (seqs_cov_dir, ti)

  if len(seqs_cov_files) == 0:
    print('Sequence coverage files not found, e.g. %s' % seqs_cov_file, file=sys.stderr)
    exit(1)

  seq_pool_len = h5py.File(seqs_cov_files[0], 'r')['targets'].shape[1]
  num_targets = len(seqs_cov_files)

  ################################################################
  # read targets

  # extend targets
  num_targets_tfr = num_targets
  if options.target_extend is not None:
    assert(options.target_extend >= num_targets_tfr)
    num_targets_tfr = options.target_extend

  # initialize targets
  targets = np.zeros((num_seqs, seq_pool_len, num_targets_tfr), dtype='float16')

  # read each target
  for ti in range(num_targets):
    seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
    tii = options.target_start + ti
    targets[:,:,tii] = seqs_cov_open['targets'][options.start_i:options.end_i,:]
    seqs_cov_open.close()
  # threshold each sequence using an arbitrary threshold
  mask_by_thr = np.any(np.any(targets > options.threshold, axis=1), axis=-1)
  idx_filt_seqs = np.argwhere(mask_by_thr).flatten()
  num_seqs_to_add = len(idx_filt_seqs)
  for i in range(5):
    print('*')
  print(num_seqs_to_add)
  for i in range(5):
    print('*')
  # current_json = open('%s/statistics.json' % options.out_dir, 'r')
  # current_stats = json.load(current_json)
  # current_stats['%s_seqs'%fold_set] += num_seqs_to_add # update number of seqs

  # with open('%s/statistics.json' % options.out_dir, 'w') as stats_json_out:
  #   json.dump(current_stats, stats_json_out, indent=4)

  count_dir = os.path.join(options.out_dir, 'counts')
  if not os.path.isdir(count_dir):
    os.mkdir(count_dir)
  file_id = fold_set+'_'+uuid.uuid4().hex
  file_path = os.path.join(count_dir, file_id)
  f = open(file_path, 'w')
  f.write(str(num_seqs_to_add))
  f.close()



  ################################################################
  # modify unmappable
  #
  # if options.umap_npy is not None and options.umap_clip < 1:
  #   unmap_mask = np.load(options.umap_npy)
  #
  #   for si in idx_filt_seqs:
  #     msi = options.start_i + si
  #
  #     # determine unmappable null value
  #     seq_target_null = np.percentile(targets[si], q=[100*options.umap_clip], axis=0)[0]
  #
  #     # set unmappable positions to null
  #     targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)
  #
  # elif options.umap_npy is not None and options.umap_tfr:
  #   unmap_mask = np.load(options.umap_npy)

  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')
  with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
    for si in idx_filt_seqs:

      msi = options.start_i + si
      mseq = model_seqs[msi]

      # read FASTA
      seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)

      # one hot code
      seq_1hot = dna_1hot(seq_dna)
      # seq_1hot = dna_1hot_index(seq_dna) # more efficient, but fighting inertia
      # hash to bytes
      features_dict = {
        'coordinate': feature_str('{}_{}_{}'.format(mseq.chr, mseq.start, mseq.end).encode()),
        # 'sequence': feature_bytes(seq_1hot),
        'sequence': feature_bytes(seq_1hot),
        'target': feature_bytes(targets[si,:,:])
        }
      # features_dict = {
      #   'chrom': feature_str(mseq.chr.encode()),
      #   # 'sequence': feature_bytes(seq_1hot),
      #   'start': feature_floats(mseq.start),
      #   'end': feature_floats(mseq.end),
      #   'sequence': feature_bytes(seq_1hot),
      #   'target': feature_bytes(targets[si,:,:])
      #   }
      # add unmappability
      if options.umap_tfr:
        features_dict['umap'] = feature_bytes(unmap_mask[msi,:])

      # write example
      example = tf.train.Example(features=tf.train.Features(feature=features_dict))
      writer.write(example.SerializeToString())

    fasta_open.close()




def feature_bytes(values):
  """Convert numpy arrays to bytes features."""
  values = values.flatten().tostring()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def feature_str(values):
  """Convert str to bytes features."""
  # value = np.array(values)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def feature_floats(values):
  """Convert numpy arrays to floats features.
     Requires more space than bytes."""
  values = values.flatten().tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
