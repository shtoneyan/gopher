#!/usr/bin/env python
import sys
sys.path.append('../gopher')
import glob
import metrics
import numpy as np
import os
import pandas as pd
import re
import sys
import tensorflow as tf
import utils
import wandb
import yaml
from scipy import stats
from tqdm import tqdm
import h5py
from collections import OrderedDict

def get_metrics_fast(h5_filename, true_mean, pred_mean, sts,
                     bin_size, batch_size=64, scaling_factors=[]):
    x_den = 0
    y_den = 0
    nom = 0
    se = 0
    if len(scaling_factors) > 0:
        pred_mean = pred_mean * scaling_factors
    with h5py.File(h5_filename, "r") as f:
        true_batch = utils.batch_np(f['true'], batch_size)
        pred_batch = utils.batch_np(f['pred'], batch_size)
        for true, pred in tqdm(zip(true_batch, pred_batch)):
            true = true.reshape(-1, sts['num_targets'])
            pred = pred.reshape(-1, sts['num_targets'])
            if len(scaling_factors) > 0:
                pred = pred * scaling_factors
            nom += np.sum((true - true_mean) * (pred - pred_mean), axis=0)
            x_den += np.sum((true - true_mean) ** 2, axis=0)
            diff_pred = np.around((pred - pred_mean)**2, 14)
            y_den += np.sum(diff_pred, axis=0)
            se += ((true - pred) ** 2).sum(axis=0)
    concatenated_pr = (nom / np.sqrt((x_den * y_den)))
    mse = se/(sts['test_seqs'] * sts['seq_length']//bin_size)
    return concatenated_pr, mse

def change_resolution(cov, bin_size_orig, eval_bin):
    """
    Decrease resolution of coverage values
    :param cov: coverage value 3D array
    :param bin_size_orig: original bin size
    :param eval_bin: bin size to re-bin it to
    :return: array with changed resolution
    """
    assert cov.ndim == 3, 'Wrong number of dims'
    assert eval_bin >= bin_size_orig, 'New bin size cannot be smaller than original!'
    N, L, C = cov.shape
    binned_cov = cov.reshape(N, L * bin_size_orig // eval_bin, eval_bin // bin_size_orig, C).mean(axis=2)
    return binned_cov


def choose_corr_func(testset_type):
    """

    :param testset_type: evaluation type, whole corresponds to concatenated, per_seq to per sequence correlation
    :return: function to calculate correlation
    """
    if testset_type == 'whole':
        get_pr = metrics.get_correlation_concatenated
    elif testset_type == 'per_seq':
        get_pr = metrics.get_correlation_per_seq
    else:
        raise ValueError
    return get_pr


def get_performance(all_truth, all_pred, targets, testset_type):
    """
    This function computes a summary of performance according to a range of metrics
    :param all_truth: ground truth in np array
    :param all_pred: predictions in np array
    :param targets: iterable of prediction target names or ids
    :param testset_type: whole or per_seq
    :return: pandas dataframe of performance values per target
    """
    assert all_truth.shape[-1] == all_pred.shape[-1], 'Incorrect number of cell lines for true and pred'
    assert testset_type == 'whole' or testset_type == 'per_seq', 'Unknown testset type'

    mse = metrics.get_mse(all_truth, all_pred).mean(axis=1).mean(axis=0)

    js_per_seq = metrics.get_js_per_seq(all_truth, all_pred).mean(axis=0)
    js_conc = metrics.get_js_concatenated(all_truth, all_pred)
    poiss = metrics.get_poiss_nll(all_truth, all_pred).mean(axis=1).mean(axis=0)
    try:
        pr_corr = choose_corr_func(testset_type)(all_truth, all_pred,
                                                 corr_type='pearsonr')
    except ValueError:
        pr_corr = [np.nan for i in range(len(poiss))]

    try:
        sp_corr = choose_corr_func(testset_type)(all_truth, all_pred,
                                                 corr_type='spearmanr')
    except ValueError:
        sp_corr = [np.nan for i in range(len(poiss))]
    performance = {'mse': mse, 'js_per_seq': js_per_seq, 'js_conc': js_conc,
                   'poiss': poiss, 'pr_corr': pr_corr, 'sp_corr': sp_corr,
                   'targets': targets}

    return pd.DataFrame(performance)


def get_scaling_factors(all_truth, all_pred):
    """
    Compute factors to scale each target prediction
    :param all_truth: ground truth
    :param all_pred: predicitons
    :return: scaling factors corresponding to each target
    """
    N, L, C = all_pred.shape
    flat_pred = all_pred.reshape(N * L, C)
    flat_truth = all_truth.reshape(N * L, C)
    truth_per_cell_line_sum = flat_truth.sum(axis=0)
    pred_per_cell_line_sum = flat_pred.sum(axis=0)
    scaling_factors = truth_per_cell_line_sum / pred_per_cell_line_sum
    return scaling_factors


def get_performance_raw_scaled(truth, targets, pred_labels, eval_type):
    """
    Calculates the performance using raw predictions and scaled predictions
    :param truth: ground truth
    :param targets: prediction targets
    :param pred_labels: dictionary of prediction type and corresponding raw or scaled predictions
    :param eval_type: 'whole' or 'per_seq' corresponding to concatenated or per sequence correlations
    :return: dataframe of performance values
    """

    complete_performance = []
    for label, pred in pred_labels.items():
        # get performance df
        performance = get_performance(truth, pred, targets, eval_type)
        performance['pred type'] = label
        performance['eval type'] = eval_type
        complete_performance.append(performance)
    return pd.concat(complete_performance)


def evaluate_run(model, bin_size, testset, targets, eval_type='whole'):
    """
    Calculates predictions, scaling factors and gets raw and scaled predictions
    :param model: h5 file path to trained model
    :param bin_size: resolution
    :param testset: test dataset
    :param targets: prediction targets
    :return: performance dataframe and sclaing factors
    """
    # make predictions
    truth, raw_pred = utils.get_true_pred(model, bin_size, testset)
    # get scales predictions
    scaling_factors = get_scaling_factors(truth, raw_pred)
    if (np.isfinite(scaling_factors)).sum() == len(scaling_factors):  # do only if all factors are ok
        scaled_pred = raw_pred * scaling_factors
        sets_to_process = OrderedDict()
        sets_to_process['raw'] = raw_pred
        sets_to_process['scaled'] = scaled_pred
        all_scaling_factors = [1 for i in range(len(scaling_factors))]
        all_scaling_factors += list(scaling_factors)
    else:
        sets_to_process = {'raw': raw_pred}
        all_scaling_factors = [1 for i in range(len(scaling_factors))]
    complete_performance = get_performance_raw_scaled(truth, targets,
                                                      sets_to_process, eval_type)

    complete_performance['scaling_factors'] = all_scaling_factors
    return complete_performance

def merge_performance_with_metadata(performance, run_dir):
    """
    Merges performance evaluation data table with run metadata
    :param performance: pandas dataframe of performance values
    :param run_dir: run directory with model and config
    :return: pandas dataframe of performance and run summary descriptions and scaling factors for each cell line
    """
    # get metadata for the run
    metadata = utils.get_run_metadata(run_dir)
    # add metadata to performance dataframes
    n_rows = performance.shape[0]
    metadata_broadcasted = pd.DataFrame(np.repeat(metadata.values, n_rows, axis=0), columns=metadata.columns)
    performance_w_metadata = pd.concat([performance.reset_index(), metadata_broadcasted], axis=1)
    performance_w_metadata['run_dir'] = run_dir
    # print('SCALING FACTORS')
    # print(scaling_factors)
    # if len(scaling_factors)>0:
    #     performance_w_metadata['scaling_factors'] = scaling_factors
    # else:
    #     performance_w_metadata['scaling_factors'] = [1 for i in range(len(scaling_factors))].append(scaling_factors)
    return performance_w_metadata

def evaluate_run_fast(run_dir, testset, targets, sts, model, bin_size, batch_size, scale):
    # load model
    tmp_file = os.path.basename(os.path.abspath(run_dir))+'.h5'
    # save truth and predictions in h5 and return true mean of each
    true_mean, pred_mean = utils.write_true_pred_to_h5(testset, sts, model, h5_filename=tmp_file, bin_size=bin_size)
    if scale:
        scaling_factors = true_mean / pred_mean
        pred_type = 'scaled'
    else:
        scaling_factors = [1 for i in range(sts['num_targets'])]
        pred_type = 'raw'
    # calculate pearson r and mse using h5
    concatenated_pr, mse = get_metrics_fast(tmp_file, true_mean, pred_mean, sts, bin_size, batch_size,
                                            scaling_factors=scaling_factors)
    os.remove(tmp_file) # clean up h5
    summary_results = pd.DataFrame({'mse': mse, 'pr_corr': concatenated_pr, 'targets': targets})  # summary per target
    summary_results['eval type'] = 'whole'
    summary_results['pred type'] = pred_type
    summary_results['scaling_factors'] = scaling_factors
    return summary_results


def process_run_list(run_dirs, output_summary_filepath, data_dir, batch_size, eval_type='whole', fast=False, scale=False):
    """
    Evaluate a set of runs
    :param run_dirs: iterable of run directories
    :param output_summary_filepath: path where to save result dataframes
    :return:
    """
    # get datasets
    testset, targets, sts = utils.collect_whole_testset(data_dir=data_dir,
                                             coords=False, batch_size=batch_size,  return_sts=True)
    # process runs
    all_run_summaries = []
    for run_dir in run_dirs:
        print(run_dir)
        # load model
        model, bin_size = utils.read_model(run_dir, compile_model=False)
        # evaluate run
        if fast:
            performance = evaluate_run_fast(run_dir, testset, targets, sts, model, bin_size,
                                                             batch_size, scale=scale)
        else:
            performance = evaluate_run(model, bin_size, testset, targets, eval_type=eval_type)
        # combine with run metadata
        run_summary = merge_performance_with_metadata(performance, run_dir)
        all_run_summaries.append(run_summary)  # save each run data into a list
    pd.concat(all_run_summaries).to_csv(output_summary_filepath, index=False)


def collect_run_dirs(project_name, wandb_dir='paper_runs/*/*/*', ignore_duplicates=False):
    """
    Collects saved directories for a given WandB project
    :param project_name: name used in the WandB run
    :param wandb_dir: directory pattern where all the runs are saved
    :return: list of run directories of a project
    """
    wandb.login()
    api = wandb.Api()
    runs = api.runs(project_name)
    run_dirs = []
    for run in runs:
        matching_run_paths = glob.glob(wandb_dir + run.id)
        if len(matching_run_paths) == 1:
            run_dirs.append(matching_run_paths[0])
        elif len(matching_run_paths) == 0:
            print(run, 'FOLDER NOT FOUND')
        else:
            if ignore_duplicates:
                print('Warning: multiple matches found')
#                 run_dirs.append([m for m in matching_run_paths if 'best_model.h5' in os.listdir(m+'/files')][0])
                run_dirs.append(matching_run_paths)
            else:
                raise Exception('too many runs match id {}'.format(run))
    return run_dirs


def collect_sweep_dirs(sweep_id, wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*'):
    """
    Collect sweep runs. This is useful if a WandB project includes failed sweeps.
    :param sweep_id: WandB sweep id
    :param wandb_dir: wandb run save directory
    :return: list of run paths
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    sweep_runs = sweep.runs
    run_dirs = [glob.glob(wandb_dir + run.id)[0] for run in sweep_runs]
    return run_dirs


def evaluate_project(data_dir, run_dir_list=None, project_dir=None, wandb_project_name=None,
                     wandb_dir=None, output_dir='output',
                     output_prefix=None, batch_size=32, eval_type='whole', fast=False, scale=False):
    """
    Evaluates a set of runs useing whole and IDR test sets, raw and scaled predictions and multiple metrics.
    :param data_dir: whole test set directory
    :param run_dir_list: list of run paths to evaluate
    :param project_dir: directory that includes all the runs to evaluate
    :param wandb_project_name: WandB project to evaluate
    :param wandb_dir: directory where runs are saved, needed if using project name
    :param output_dir: directory where to save the result
    :param output_prefix: file name for the result
    :param batch_size: batch size for evaluation (set to smaller for bigger models)
    :param eval_type: whole or per_seq for concatenated or per sequence evaluation
    :return: None
    """
    assert run_dir_list or project_dir or wandb_project_name, 'Must provide a list of runs, a project dir or WandB ' \
                                                              'project name! '
    utils.make_dir(output_dir)  # create output directory
    # option 1: project directory is provided with run outputs all of which should be evaluated
    if os.path.isdir(str(project_dir)):  # check if dir exists
        # get all subdirs that have model saved
        all_run_dirs = [g for g in glob.glob(project_dir + '/**/run*', recursive=True) if os.path.isdir(g)]
        run_dir_list = [os.path.join(project_dir, d) for d in all_run_dirs
                        if os.path.isfile(os.path.join(d, 'files/best_model.h5'))]
        if not output_prefix:
            output_prefix = os.path.basename(project_dir.rstrip('/'))  # if no project name use run dir name

        print('SELECTED ALL RUNS IN DIRECTORY: ' + project_dir)
    # option 2: use a list of predefined run paths
    elif run_dir_list:
        if not output_prefix:
            output_prefix = 'evaluation_results'
        print('USING PREDEFINED LIST OF RUNS')

    # option 3: use WandB project name
    else:
        assert os.path.isdir(str(wandb_dir)), 'WandB output directory not found!'
        run_dir_list = collect_run_dirs(wandb_project_name, wandb_dir=wandb_dir)
        if not output_prefix:
            output_prefix = wandb_project_name
        print('COLLECTING RUNS FROM PROJECT IN WANDB')

    assert run_dir_list, 'No run paths found'
    csv_filename = output_prefix + '.csv'  # filename
    result_path = os.path.join(output_dir, csv_filename)  # output path
    print(len(run_dir_list))
    # process a list of runs for evaluation
    process_run_list(run_dir_list, result_path, data_dir, batch_size=batch_size, eval_type=eval_type, fast=fast, scale=scale)
