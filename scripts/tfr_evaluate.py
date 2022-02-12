#!/usr/bin/env python
import tensorflow as tf
import wandb
import glob, os, sys
import yaml
import pandas as pd
import numpy as np
from scipy import stats
import util
import metrics

def open_bw(bw_filename, chrom_size_path):
    '''This function opens a new bw file'''
    assert not os.path.isfile(bw_filename), 'Bw at {} alread exists!'.format(bw_filename)
    chrom_sizes = read_chrom_size(chrom_size_path) # load chromosome sizes
    bw = pyBigWig.open(bw_filename, "w") # open bw
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()], maxZooms=0)
    return bw # bw file

def get_vals_per_range(bw_path, bed_path):
    '''This function reads bw (specific ranges of bed file) into numpy array'''
    bw = pyBigWig.open(bw_path)
    bw_list = []
    for line in open(bed_path):
        cols = line.strip().split()
        vals = bw.values(cols[0], int(cols[1]), int(cols[2]))
        bw_list.append(vals)
    bw.close()
    return bw_list

def get_config(run_path):
    '''This function returns config of a wandb run as a dictionary'''
    config_file = os.path.join(run_path, 'files', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_model(run_path, compile_model=True):
    '''This function loads a per-trained model'''
    config = get_config(run_path) # load wandb config
    if 'bin_size' in config.keys():
        bin_size = config['bin_size']['value'] # get bin size
    else:
        bin_size = 'NA'
    model_path = os.path.join(run_path, 'files', 'best_model.h5') # pretrained model
    # load model
    trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    if compile_model:
        loss_fn_str = config['loss_fn']['value'] # get loss
        loss_fn = eval(loss_fn_str)() # turn loss into function
        trained_model.compile(optimizer="Adam", loss=loss_fn)
    return trained_model, bin_size # model and bin size

def describe_run(run_path, columns_of_interest=['model_fn', 'bin_size', 'crop', 'rev_comp']):
    metadata = tfr_evaluate.get_run_metadata(run_path)
    if 'data_dir' in metadata.columns:
        model_id = [metadata['data_dir'].values[0].split('/i_3072_w_1')[0].split('/')[-1]]
    else:
        model_id = []
    for c in columns_of_interest:
        if c in metadata.columns:
            model_id.append(str(metadata[c].values[0]))
    return ' '.join(model_id)

def get_true_pred(model, bin_size, testset):
    # model, bin_size = read_model(run_path, compile_model=False)
    all_truth = []
    all_pred = []
    for i, (x, y) in enumerate(testset):
        p = model.predict(x)
        binned_y = util.bin_resolution(y, bin_size)
        y = binned_y.numpy()
        all_truth.append(y)
        all_pred.append(p)
    return np.concatenate(all_truth), np.concatenate(all_pred)

def change_resolution(truth, bin_size_orig, eval_bin):
    N, L, C  = truth.shape
    binned_truth = truth.reshape(N, L*bin_size_orig//eval_bin, eval_bin//bin_size_orig, C).mean(axis=2)
    return (binned_truth)

def split_into_2k_chunks(x, input_size=2048):
    N = tf.shape(x)[0]
    L = tf.shape(x)[1]
    C = tf.shape(x)[2]
    x_4D = tf.reshape(x, (N, L//input_size, input_size, C))
    x_split_to_2k = tf.reshape(x_4D, (N*L//input_size, input_size, C))
    return x_split_to_2k

def combine_into_6k_chunks(x, chunk_number=3):
    N, L, C = x.shape
    x_6k = np.reshape(x, (N//chunk_number, chunk_number*L, C))
    return x_6k

def choose_corr_func(testset_type):
    if testset_type == 'whole':
        get_pr = metrics.get_correlation_concatenated
    elif testset_type == 'idr':
        get_pr = metrics.get_correlation_per_seq
    return get_pr

def get_performance(all_truth, all_pred, targets, testset_type):
    assert all_truth.shape[-1] == all_pred.shape[-1], 'Incorrect number of cell lines for true and pred'
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
                    'poiss': poiss, 'pr_corr': pr_corr, 'sp_corr':sp_corr,
                    'targets':targets}
    return pd.DataFrame(performance)

def get_scaling_factors(all_truth, all_pred):
    N, L, C = all_pred.shape
    flat_pred = all_pred.reshape(N*L, C)
    flat_truth = all_truth.reshape(N*L, C)
    truth_per_cell_line_sum = flat_truth.sum(axis=0)
    pred_per_cell_line_sum = flat_pred.sum(axis=0)
    scaling_factors =  truth_per_cell_line_sum / pred_per_cell_line_sum
    return scaling_factors

def get_performance_raw_scaled(truth, targets, pred_labels, eval_type):
    complete_performance = []
    for label, pred in pred_labels.items():
        # get performance df
        performance = get_performance(truth, pred, targets, eval_type)
        performance['pred type'] = label
        performance['eval type'] = eval_type
        complete_performance.append(performance)
    return pd.concat(complete_performance)


def evaluate_run_whole(model, bin_size, testset, targets):
    # make predictions
    truth, raw_pred = get_true_pred(model, bin_size, testset)
    # get scales predictions
    scaling_factors = get_scaling_factors(truth, raw_pred)
    if (np.isfinite(scaling_factors)).sum() == len(scaling_factors): # if all factors are ok
        scaled_pred = raw_pred * scaling_factors
        sets_to_process = {'raw': raw_pred, 'scaled': scaled_pred}
    else:
        sets_to_process = {'raw': raw_pred}
    complete_performance = get_performance_raw_scaled(truth, targets,
                                                      sets_to_process, 'whole')
    return (complete_performance, scaling_factors)

def extract_datasets(path_pattern='/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/15_IDR_test_sets_6K/cell_line_*/i_6144_w_1/'):
    paths = glob.glob(path_pattern)
    assert len(paths)>0
    target_dataset = {}
    for path in paths:
        sts = util.load_stats(path)
        testset_6K = util.make_dataset(path, 'test', sts, batch_size=512, shuffle=False)
        target = pd.read_csv(path+'targets.txt', sep='\t')['identifier'].values[0]
        i = [f for f in path.split('/') if 'cell_line' in f][0].split('_')[-1]
        testset_2K = testset_6K.map(lambda x,y: (split_into_2k_chunks(x), split_into_2k_chunks(y)))
        target_dataset[(int(i), target)] = testset_2K
    return target_dataset

def evaluate_run_idr(model, bin_size, target_dataset, scaling_factors):
    complete_performance = []
    for (i, target), one_testset in target_dataset.items():
        # make predictions and slice the cell line
        truth, all_pred = get_true_pred(model, bin_size, one_testset)
        raw_pred = np.expand_dims(all_pred[:,:,i], axis=-1)
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
    config = get_config(run_dir)
    relevant_config = {k:[config[k]['value']] for k in config.keys() if k not in ['wandb_version', '_wandb']}
    metadata = pd.DataFrame(relevant_config)
    metadata['run_dir'] = run_dir
    return metadata

def collect_whole_testset(data_dir='/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/', coords=False, batch_size=512):
    sts = util.load_stats(data_dir)
    testset = util.make_dataset(data_dir, 'test', sts, batch_size=batch_size, shuffle=False, coords=coords)
    targets = pd.read_csv(data_dir+'targets.txt', sep='\t')['identifier'].values
    return testset, targets

def collect_datasets(data_dir='/home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/'):
    # get testset
    testset, targets = collect_whole_testset()
    # get cell line specific IDR testsets in 6K
    target_dataset_idr = extract_datasets()
    return (testset, targets, target_dataset_idr)

def evaluate_run_whole_idr(run_dir, testset, targets, target_dataset_idr):
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
    bad_runs = []
    for run_dir in run_dirs:
        model_path = os.path.join(run_dir, 'files', 'best_model.h5')
        if not os.path.isfile(model_path):
            bad_runs.append(run_dir)
            print('No saved model found, skipping run at '+ run_dir)
    if len(bad_runs)>0:
        util.writ_list_to_file(bad_runs, error_output_filepath)
    return bad_runs


def process_run_list(run_dirs, output_summary_filepath):
    # get datasets
    testset, targets, target_dataset_idr = collect_datasets()
    #check runs
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
    if len(all_run_summaries)>0:
        pd.concat(all_run_summaries).to_csv(output_summary_filepath, index=False)
        pd.concat(all_scale_summaries).to_csv(output_summary_filepath.replace('.csv', '_SCALES.csv'), index=False)
    else:
        print('No runs with saved models found!')

def collect_run_dirs(project_name, wandb_dir='paper_runs/*/*/*'):
    wandb.login()
    api = wandb.Api()
    runs = api.runs(project_name)
    run_dirs = [glob.glob(wandb_dir+run.id)[0] for run in runs]
    return run_dirs

def collect_sweep_dirs(sweep_id, wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*'):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    sweep_runs = sweep.runs
    run_dirs = [glob.glob(wandb_dir+run.id)[0] for run in sweep_runs]
    return run_dirs


if __name__ == '__main__':
    run_dirs = []
    # dir_of_all_runs = '/home/shush/profile/QuantPred/paper_runs/basenji/augmentation_basenji'
    dir_of_all_runs = sys.argv[1]
    output_dir = 'colab_eval' # output dir
    util.make_dir(output_dir)
    # project name in wandb or name to use for saving if list of runs provided
    project_name = 'COLAB_MODEL_SELECTION'

    testset, targets, target_dataset_idr = collect_datasets()
    # if pre-assembled directory of runs given then take all
    if os.path.isdir(dir_of_all_runs):
        run_dirs = [os.path.join(dir_of_all_runs, d) for d in os.listdir(dir_of_all_runs)
                    if os.path.isfile(os.path.join(dir_of_all_runs, d, 'files/best_model.h5'))]
        project_name = os.path.basename(dir_of_all_runs.rstrip('/'))
        assert len(project_name)>0, 'Invalid project name'
        print('SELECTED ALL RUNS IN DIRECTORY: ' + dir_of_all_runs)
        print('PROJECT NAME: ' + project_name)

    # else check if list of runs also is absent then collect runs
    elif len(run_dirs) == 0:
        run_dirs = collect_run_dirs(project_name, wandb_dir='/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/*/*')
        print('COLLECTING RUNS FROM PROJECT IN WANDB')
        print(run_dirs)
    else:
        print('USING PREDEFINED LIST OF RUNS')
    csv_filename = project_name + '.csv'
    result_path = os.path.join(output_dir, csv_filename)
    print(result_path)
    process_run_list(run_dirs, result_path)
