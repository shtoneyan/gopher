import os
import numpy as np
import pandas as pd
from scipy import stats
import shutil
import seaborn as sns
import util
import metrics
import yaml
import wandb
import tfr_evaluate
from tqdm import tqdm
# from test_to_bw_fast import get_config

def smoothen(true_2K, pred_2K, eval_smooth_window_size):
    N, L, C = true_2K.shape
    smooth_true = np.zeros((N*L, C))
    smooth_pred = np.zeros((N*L, C))
    kernel = np.ones(eval_smooth_window_size) / eval_smooth_window_size
    for cell_line in range(C):
        smooth_true[:,cell_line] = np.convolve(true_2K[:,:,cell_line].flatten(), kernel, mode='same')
        smooth_pred[:,cell_line] = np.convolve(pred_2K[:,:,cell_line].flatten(), kernel, mode='same')
    return (smooth_true.reshape(N,L,C), smooth_pred.reshape(N,L,C))

bin_run = {}
for run_dir in tfr_evaluate.collect_run_dirs('BASENJI_BIN_LOSS'):
    parameters = tfr_evaluate.get_config(run_dir)
    if parameters['loss_fn']['value'] == 'poisson':
        bin_run[parameters['bin_size']['value']] = run_dir

testset, targets = tfr_evaluate.collect_whole_testset()


# get performance metrics for various evaluation bin sizes
bin_sizes = np.array([1, 32, 64, 128, 256, 512, 1024, 2048])
complete_perf = []
for raw_bin_size in tqdm(bin_sizes):
    model, bin_size = tfr_evaluate.read_model(bin_run[raw_bin_size], compile_model=False)
    assert bin_size == raw_bin_size, 'Error in bin size path dictionary!'
    all_true, all_pred = tfr_evaluate.get_true_pred(model, bin_size, testset)
    true_2K = np.repeat(all_true, bin_size, axis=1)
    pred_2K = np.repeat(all_pred, bin_size, axis=1)
    for eval_smooth_window_size in bin_sizes[bin_sizes>=raw_bin_size]:
        print(raw_bin_size, '--->', eval_smooth_window_size)
        if eval_smooth_window_size > raw_bin_size: # smoothen if bin size bigger than original
            true_for_eval, pred_for_eval = smoothen(true_2K, pred_2K, eval_smooth_window_size)
        else:
            true_for_eval, pred_for_eval = true_2K, pred_2K
        perf_avg = tfr_evaluate.get_performance(true_for_eval, pred_for_eval, targets, 'whole').mean()
        one_resolution_result = [raw_bin_size, eval_smooth_window_size] + list(perf_avg.values)
        column_names = ['raw bin size', 'smooth bin size'] + list(perf_avg.index)
        df = pd.DataFrame(one_resolution_result).T
        df.columns=column_names
        complete_perf.append(df)

complete_result_dataset = pd.concat(complete_perf)

complete_result_dataset.to_csv('smoothen_results_basenji.csv')
