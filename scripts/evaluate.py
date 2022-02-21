#!/usr/bin/env python
import tensorflow as tf
import wandb
import glob, os, sys
import yaml
import pandas as pd
import numpy as np
from scipy import stats
import utils
import metrics
import re


def open_bw(bw_filename, chrom_size_path):
    '''
    This function opens a new bw file
    :param bw_filename: path to bw file to be created
    :param chrom_size_path: chrom size file for corresponding genome assembly
    :return: bw object
    '''
    assert not os.path.isfile(bw_filename), 'Bw at {} alread exists!'.format(bw_filename)
    chrom_sizes = read_chrom_size(chrom_size_path)  # load chromosome sizes
    bw = pyBigWig.open(bw_filename, "w")  # open bw
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()], maxZooms=0)
    return bw  # bw file


def get_vals_per_range(bw_path, bed_path):
    '''
    This function reads bw (specific ranges of bed file) into numpy array
    :param bw_path: existing bw file path
    :param bed_path: bed file path to read the coordinates from
    :return: list of coverage values that can be of different lengths
    '''
    bw = pyBigWig.open(bw_path)
    bw_list = []
    for line in open(bed_path):
        cols = line.strip().split()
        vals = bw.values(cols[0], int(cols[1]), int(cols[2]))
        bw_list.append(vals)
    bw.close()
    return bw_list


def get_config(run_path):
    '''
    This function returns config of a wandb run as a dictionary
    :param run_path: dir with run outputs
    :return: dictionary of configs
    '''
    config_file = os.path.join(run_path, 'files', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_model(run_path, compile_model=True):
    '''
    This function loads a per-trained model
    :param run_path: run output dir
    :param compile_model: bool compile model using loss from config
    :return: model and resolution
    '''
    config = get_config(run_path)  # load wandb config
    if 'bin_size' in config.keys():
        bin_size = config['bin_size']['value']  # get bin size
    else:
        bin_size = 'NA'
    model_path = os.path.join(run_path, 'files', 'best_model.h5')  # pretrained model
    # load model
    trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    if compile_model:
        loss_fn_str = config['loss_fn']['value']  # get loss
        loss_fn = eval(loss_fn_str)()  # turn loss into function
        trained_model.compile(optimizer="Adam", loss=loss_fn)
    return trained_model, bin_size  # model and bin size


def describe_run(run_path, columns_of_interest=['model_fn', 'bin_size', 'crop', 'rev_comp']):
    """
    Get the run descriptors from config
    :param run_path: output from training
    :param columns_of_interest: entries in the config file that need to be extracted
    :return: str decription of run
    """
    metadata = get_run_metadata(run_path)
    model_id = []
    if 'data_dir' in metadata.columns:
        p = re.compile('i_[0-9]*_w_1')
        dataset_subdir = p.search(metadata['data_dir'].values[0])
        if dataset_subdir:
            model_id = [metadata['data_dir'].values[0].split('/' + dataset_subdir.group(0))[0].split('/')[-1]]
    for c in columns_of_interest:
        if c in metadata.columns:
            model_id.append(str(metadata[c].values[0]))
    return ' '.join(model_id)


def get_true_pred(model, bin_size, testset):
    """
    Iterate through dataset and get predictions into np
    :param model: model path to h5
    :param bin_size: resolution
    :param testset: tf dataset or other iterable
    :return: np arrays of ground truth and predictions
    """
    all_truth = []
    all_pred = []
    for i, (x, y) in enumerate(testset):
        p = model.predict(x)
        binned_y = utils.bin_resolution(y, bin_size)  # change resolution
        y = binned_y.numpy()
        all_truth.append(y)
        all_pred.append(p)
    return np.concatenate(all_truth), np.concatenate(all_pred)


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
    return (binned_cov)


def split_into_2k_chunks(x, input_size=2048):
    """
    Split into smaller parts
    :param x: sequence onehot tf tensor
    :param input_size: split size
    :return: split tensor
    """
    N = tf.shape(x)[0]
    L = tf.shape(x)[1]
    C = tf.shape(x)[2]
    x_4D = tf.reshape(x, (N, L // input_size, input_size, C))
    x_split_to_2k = tf.reshape(x_4D, (N * L // input_size, input_size, C))
    return x_split_to_2k


def combine_into_6k_chunks(x, chunk_number=3):
    """
    Reassemble into bigger chunks
    :param x: onehot sequence tf tensor
    :param chunk_number: number of units to combine
    :return: combined tensor
    """
    N, L, C = x.shape
    x_6k = np.reshape(x, (N // chunk_number, chunk_number * L, C))
    return x_6k


def choose_corr_func(testset_type):
    """

    :param testset_type: evaluation type, whole corresponds to concatenated, idr to per sequence correlation
    :return: function to calculate correlation
    """
    if testset_type == 'whole':
        get_pr = metrics.get_correlation_concatenated
    elif testset_type == 'idr':
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
    :param testset_type: whole or idr
    :return: pandas dataframe of performance values per target
    """
    assert all_truth.shape[-1] == all_pred.shape[-1], 'Incorrect number of cell lines for true and pred'
    assert testset_type == 'whole' or testset_type == 'idr', 'Unknown testset type'
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
    :param eval_type: 'whole' or 'idr' corresponding to concatenated or per sequence correlations
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


def evaluate_run_whole(model, bin_size, testset, targets):
    """
    Calculates predictions, scaling factors and gets raw and scaled predictions
    :param model: h5 file path to trained model
    :param bin_size: resolution
    :param testset: test dataset
    :param targets: prediction targets
    :return: performance dataframe and sclaing factors
    """
    # make predictions
    truth, raw_pred = get_true_pred(model, bin_size, testset)
    # get scales predictions
    scaling_factors = get_scaling_factors(truth, raw_pred)
    if (np.isfinite(scaling_factors)).sum() == len(scaling_factors):  # do only if all factors are ok
        scaled_pred = raw_pred * scaling_factors
        sets_to_process = {'raw': raw_pred, 'scaled': scaled_pred}
    else:
        sets_to_process = {'raw': raw_pred}
    complete_performance = get_performance_raw_scaled(truth, targets,
                                                      sets_to_process, 'whole')
    return (complete_performance, scaling_factors)


def extract_IDR_datasets(
        path_pattern='/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/15_IDR_test_sets_6K/cell_line_*/i_6144_w_1/'):
    """
    get cell line specific IDR datasets from multiple subdirectories
    :param path_pattern: pattern that includes each subdir of cell lines
    :return: dictionary of idr datasets
    """
    paths = glob.glob(path_pattern)
    assert len(paths) > 0
    target_dataset = {}
    for path in paths:
        sts = utils.load_stats(path)
        testset_6K = utils.make_dataset(path, 'test', sts, batch_size=512, shuffle=False)
        target = pd.read_csv(path + 'targets.txt', sep='\t')['identifier'].values[0]
        i = [f for f in path.split('/') if 'cell_line' in f][0].split('_')[-1]
        testset_2K = testset_6K.map(lambda x, y: (split_into_2k_chunks(x), split_into_2k_chunks(y)))
        target_dataset[(int(i), target)] = testset_2K
    return target_dataset


def evaluate_run_idr(model, bin_size, target_dataset, scaling_factors):
    """
    Evaluate a run only based on cell line specific IDR datasets. This is not the same as peak centered.
    In IDR only we define a separate test set for each cell line evaluation that only includes regions defined in the
    IDR bed file for that cell line. In contrast in a peak-centered dataset there are regions from cell lines that are not
    in the IDR bed file of that cell line and all cell lines have the same testset for evaluation.
    :param model: h5 model path
    :param bin_size: resolution
    :param target_dataset: target testsets in a dictionary
    :param scaling_factors: scaling for scaled predictions
    :return: dataframe of performance values for IDR evaluation
    """
    complete_performance = []
    for (i, target), one_testset in target_dataset.items():
        # make predictions and slice the cell line
        truth, all_pred = get_true_pred(model, bin_size, one_testset)
        raw_pred = np.expand_dims(all_pred[:, :, i], axis=-1)
        truth_6k = combine_into_6k_chunks(truth)
        raw_pred_6k = combine_into_6k_chunks(raw_pred)
        # make scaled predictions
        scaled_pred = raw_pred_6k * scaling_factors[i]
        # get idr performance raw, scaled
        assert truth_6k.shape == raw_pred_6k.shape, 'shape mismatch!'
        complete_performance.append(get_performance_raw_scaled(truth_6k, [target], {'raw': raw_pred_6k,
                                                                                    'scaled': scaled_pred},
                                                               'idr'))

    return pd.concat(complete_performance)


def get_run_metadata(run_dir):
    """
    Collects the metadata file of a run
    :param run_dir: directory where run is saved
    :return: dataframe of metadata run descriptors
    """
    config = get_config(run_dir)
    relevant_config = {k: [config[k]['value']] for k in config.keys() if k not in ['wandb_version', '_wandb']}
    metadata = pd.DataFrame(relevant_config)
    metadata['run_dir'] = run_dir
    return metadata


def collect_whole_testset(data_dir='/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/',
                          coords=False, batch_size=512):
    """
    Collects a test fold of a given testset without shuffling it
    :param data_dir: testset directory
    :param coords: bool indicating if coordinates should be taken
    :param batch_size: batch size, important to set to smaller number for inference on large models
    :return:
    """
    sts = utils.load_stats(data_dir)
    testset = utils.make_dataset(data_dir, 'test', sts, batch_size=batch_size, shuffle=False, coords=coords)
    targets = pd.read_csv(data_dir + 'targets.txt', sep='\t')['identifier'].values
    return testset, targets


def collect_datasets(data_dir='/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/',
                     idr_data_dir_pattern='/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/15_IDR_test_sets_6K/cell_line_*/i_6144_w_1/'):
    """
    Collects whole and IDR specific testsets
    :param data_dir: whole testset directory
    :param idr_data_dir_pattern: pattern for getting cell line specific IR datasets
    :return: whole testset, all prediction targets, dictionary of  cell line specific IDR testsets
    """
    # get testset
    testset, targets = collect_whole_testset(data_dir)
    # get cell line specific IDR testsets in 6K
    target_dataset_idr = extract_IDR_datasets(idr_data_dir_pattern)
    return (testset, targets, target_dataset_idr)


def evaluate_run_whole_idr(run_dir, testset, targets, target_dataset_idr):
    """
    Calculates whole and idr test set evaluation
    :param run_dir: run directory with model and config
    :param testset: whole testset
    :param targets: prediction targets
    :param target_dataset_idr: cell line specific IDR datasets
    :return: pandas dataframe of performance and run summary descriptions and scaling factors for each cell line
    """
    # load model
    model, bin_size = read_model(run_dir, compile_model=False)
    # get performance for the whole chromosome
    complete_performance_whole, scaling_factors = evaluate_run_whole(model, bin_size, testset, targets)
    # get performance for the IDR regions only
    complete_performance_idr = evaluate_run_idr(model, bin_size, target_dataset_idr, scaling_factors)
    # get metadata for the run
    metadata = get_run_metadata(run_dir)
    # add metadata to performance dataframes
    combined_performance = pd.concat([complete_performance_whole, complete_performance_idr]).reset_index()
    n_rows = combined_performance.shape[0]
    metadata_broadcasted = pd.DataFrame(np.repeat(metadata.values, n_rows, axis=0), columns=metadata.columns)
    combined_performance_w_metadata = pd.concat([combined_performance, metadata_broadcasted], axis=1)
    combined_performance_w_metadata['run_dir'] = run_dir
    # save scaling factors
    scaling_factors_per_cell = pd.DataFrame(zip(targets, scaling_factors,
                                                [run_dir for i in range(len(scaling_factors))]))
    return (combined_performance_w_metadata, scaling_factors_per_cell)


def check_best_model_exists(run_dirs, error_output_filepath):
    """
    Check if saved model exists
    :param run_dirs: set of run directories
    :param error_output_filepath: file where to log run paths without model saved
    :return: list of bad runs
    """
    bad_runs = []
    for run_dir in run_dirs:
        model_path = os.path.join(run_dir, 'files', 'best_model.h5')
        if not os.path.isfile(model_path):
            bad_runs.append(run_dir)
            print('No saved model found, skipping run at ' + run_dir)
    if len(bad_runs) > 0:
        utils.writ_list_to_file(bad_runs, error_output_filepath)
    return bad_runs


def process_run_list(run_dirs, output_summary_filepath, data_dir, idr_data_dir_pattern):
    """
    Evaluate a set of runs
    :param run_dirs: iterable of run directories
    :param output_summary_filepath: path where to save result dataframes
    :return:
    """
    # get datasets
    testset, targets, target_dataset_idr = collect_datasets(data_dir, idr_data_dir_pattern)
    # check runs
    bad_runs = check_best_model_exists(run_dirs, output_summary_filepath.replace('.csv', '_ERROR.txt'))
    # process runs
    all_run_summaries = []
    all_scale_summaries = []
    for run_dir in run_dirs:
        if run_dir not in bad_runs:
            print(run_dir)
            run_summary, scale_summary = evaluate_run_whole_idr(run_dir, testset, targets, target_dataset_idr)
            all_run_summaries.append(run_summary)
            all_scale_summaries.append(scale_summary)
    if len(all_run_summaries) > 0:
        pd.concat(all_run_summaries).to_csv(output_summary_filepath, index=False)
        pd.concat(all_scale_summaries).to_csv(output_summary_filepath.replace('.csv', '_SCALES.csv'), index=False)
    else:
        print('No runs with saved models found!')


def collect_run_dirs(project_name, wandb_dir='paper_runs/*/*/*'):
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
            raise Exception('too many runs match id')

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


def evaluate_project(data_dir, idr_data_dir_pattern, run_dir_list=None, project_dir=None, wandb_project_name=None, wandb_dir=None, output_dir='output',
                     output_prefix=None):
    utils.make_dir(output_dir)  # create output directory
    # option 1: project directory is provided with run outputs all of which should be evaluated
    if os.path.isdir(str(project_dir)):  # check if dir exists
        # get all subdirs that have model saved
        run_dir_list = [os.path.join(project_dir, d) for d in os.listdir(project_dir)
                    if os.path.isfile(os.path.join(project_dir, d, 'files/best_model.h5'))]
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
        try:  # see if WandB project can be found
            assert os.path.isdir(str(wandb_dir)), 'WandB output directory not found!'
            run_dir_list = collect_run_dirs(wandb_project_name, wandb_dir=wandb_dir)
            if not output_prefix:
                output_prefix = wandb_project_name
            print('COLLECTING RUNS FROM PROJECT IN WANDB')
        except ValueError:  # if project name not found throw an exception
            raise Exception('Must provide run path list, output directory or WandB project name!')
    assert run_dir_list, 'No run paths found'
    csv_filename = output_prefix + '.csv'  # filename
    result_path = os.path.join(output_dir, csv_filename)  # output path
    # process a list of runs for evaluation
    process_run_list(run_dir_list, result_path, data_dir, idr_data_dir_pattern)
