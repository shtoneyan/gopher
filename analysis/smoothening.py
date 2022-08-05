import evaluate
import pandas as pd
import sys
import glob
sys.path.append('../gopher')
import utils
import numpy as np
from tqdm import tqdm

def smoothen(true, pred, eval_smooth_window_size):
    """
    function used to smoothen ground truth and predictions
    :param true: ground truth for entire test set
    :param pred: predictions for entire test set
    :param eval_smooth_window_size: window size for average smoothing
    :return: smoothened ground truth and predictions
    """
    N, L, C = true.shape
    smooth_true = np.zeros((N * L, C))
    smooth_pred = np.zeros((N * L, C))
    kernel = np.ones(eval_smooth_window_size) / eval_smooth_window_size
    for cell_line in range(C):
        smooth_true[:, cell_line] = np.convolve(true[:, :, cell_line].flatten(), kernel, mode='same')
        smooth_pred[:, cell_line] = np.convolve(pred[:, :, cell_line].flatten(), kernel, mode='same')
    return smooth_true.reshape(N, L, C), smooth_pred.reshape(N, L, C)


# create a dictionary with bin size as key and run directory as value
bin_run = {}
for run_dir in glob.glob('../trained_models/basenji_v2/binloss_basenji_v2/run*'):
    parameters = utils.get_config(run_dir)
    if parameters['loss_fn']['value'] == 'poisson':
        bin_run[parameters['bin_size']['value']] = run_dir

# get the test set
testset, targets = utils.collect_whole_testset('../datasets/quantitative_data/testset/')

# get performance metrics for various evaluation bin sizes
bin_sizes = np.array([1, 32, 64, 128, 256, 512, 1024, 2048])
complete_perf = []
for raw_bin_size in tqdm(bin_sizes):
    model, bin_size = utils.read_model(bin_run[raw_bin_size], compile_model=False)  # read model and bin size
    assert bin_size == raw_bin_size, 'Error in bin size path dictionary!'
    all_true, all_pred = utils.get_true_pred(model, bin_size, testset)  # get all ground truth and predictions as np
    true_2K = np.repeat(all_true, bin_size, axis=1)  # expand to 2K
    pred_2K = np.repeat(all_pred, bin_size, axis=1)
    for eval_smooth_window_size in bin_sizes[bin_sizes >= raw_bin_size]:
        print(raw_bin_size, '--->', eval_smooth_window_size)
        if eval_smooth_window_size > raw_bin_size:  # smoothen if bin size bigger than original
            true_for_eval, pred_for_eval = smoothen(true_2K, pred_2K, eval_smooth_window_size)
        else:  # otherwise just return the values with no smoothing
            true_for_eval, pred_for_eval = true_2K, pred_2K
        perf_avg = evaluate.get_performance(true_for_eval, pred_for_eval, targets,
                                            'whole').mean()  # get average performance
        # compile results into df
        one_resolution_result = [raw_bin_size, eval_smooth_window_size] + list(perf_avg.values)
        column_names = ['raw bin size', 'smooth bin size'] + list(perf_avg.index)
        df = pd.DataFrame(one_resolution_result).T
        df.columns = column_names
        complete_perf.append(df)

complete_result_dataset = pd.concat(complete_perf)

complete_result_dataset.to_csv('inter_results/smoothen_basenji_v2.csv')
