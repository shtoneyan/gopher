import sys
import tensorflow as tf
import h5py, os, yaml
import umap.umap_ as umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy import stats
import pandas as pd
import subprocess
from scipy.stats import pearsonr
from tqdm import tqdm
import glob
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import tfr_evaluate, util
from test_to_bw_fast import read_model, get_config
import explain
import embed
import metrics
import quant_GIA
from optparse import OptionParser
from modelzoo import GELU


def main():
    usage = 'usage: %prog [options] <motifs> <cell_line>'
    parser = OptionParser(usage)

    parser.add_option('-o', dest='out_dir',
        default='seamonster/occlude_GIA_fin',
        help='Output directory [Default: %default]')
    parser.add_option('-n','--n_background', dest='n_background',
        default=1000, type='int',
        help='Sample number for background [Default: %default]')
    parser.add_option('--logits', dest='logits',
        default=False, action='store_true',
        help='take logits [Default: %default]')
    (options, args) = parser.parse_args()
    if len(args) != 4:
      parser.error('Must provide motifs and cell line.')
    else:
      run_path = args[0]
      motif_cluster = args[1].split(',')
      background_model = args[2]
      cell_line_name = args[3]

    print('Processing')
    print(motif_cluster)
    # load and get model layer
    # run_path = 'paper_runs/new_models/32_res/run-20211023_095131-w6okxt01'
    model = tf.keras.models.load_model(run_path, custom_objects={"GELU": GELU})
    util.make_dir(options.out_dir)
    # load and threshold data
    testset, targets = tfr_evaluate.collect_whole_testset(coords=True)
    C, X, Y = util.convert_tfr_to_np(testset, 3)
    run_name = [p for p in run_path.split('/') if 'run-' in p][0]
    gia_occ_dir = util.make_dir(os.path.join(options.out_dir, run_name))
    base_dir = util.make_dir(os.path.join(gia_occ_dir, '{}_{}'.format(cell_line_name, args[1])))
    output_dir = util.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, options.n_background)))
    # for each element in the cluster of 2 and both together
    # base_dir = util.make_dir(os.path.join(gia_add_dir, '{}_{}'.format(cell_line_name, motif)))
    if background_model == 'dinuc':
        X_set = quant_GIA.select_set('all_threshold', C, X, Y)
    elif background_model == 'none':
        X_set = quant_GIA.select_set('cell_low', C, X, Y, cell_line=np.argwhere(targets==cell_line_name)[0][0])

    gi = quant_GIA.GlobalImportance(model, targets)
    if len(motif_cluster) >1:
        combo_list = [motif_cluster] + [[m] for m in motif_cluster]
    else:
        combo_list = motif_cluster
    for each_element in combo_list:
        print(each_element)
        gi.occlude_all_motif_instances(X_set, each_element, func='mean',
                                      num_sample=options.n_background)
    df = pd.concat(gi.summary_remove_motifs)
    file_prefix = '{}_in_{}_{}'.format(args[1], cell_line_name, background_model)
    df.to_csv(os.path.join(output_dir, file_prefix+'.csv'), index=None)

################################################################################
if __name__ == '__main__':
  main()
