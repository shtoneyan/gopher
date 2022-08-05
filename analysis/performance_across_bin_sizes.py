import evaluate
import pandas as pd
import sys
import glob
sys.path.append('../gopher')
import utils
import numpy as np
import json


def get_runs(glob_pattern):
    bin_run = {}
    for run_dir in glob.glob(glob_pattern):
        config = utils.get_config(run_dir)
        if config['loss_fn']['value'] == 'poisson':
            bin_run[config['bin_size']['value']] = run_dir
    return bin_run

result_base_dir = utils.make_dir('inter_results')
# get datasets
testset, targets = utils.collect_whole_testset('../datasets/quantitative_data/testset/')

model_run_pattern = {'binloss_basenji_v2': '../trained_models/basenji_v2/binloss_basenji_v2/run*',
                     'bpnet_bin_loss_40': '../trained_models/bpnet/bin_loss_40/run*'}
for model_label, run_pattern in model_run_pattern.items():
    bin_run = get_runs(run_pattern)
    # get performance metrics for various evaluation bin sizes
    result_path = result_base_dir + '/{}_triangle_plot.txt'.format(model_label)
    bin_sizes = sorted(list(bin_run.keys()))
    performance_per_resolution = []
    for raw_bin_size in bin_sizes:
        model, _ = utils.read_model(bin_run[raw_bin_size])
        all_true, all_pred = utils.get_true_pred(model, raw_bin_size, testset)
        for eval_bin_size in bin_sizes:
            if eval_bin_size >= raw_bin_size:
                print(raw_bin_size, '--->', eval_bin_size)
                true_for_eval = evaluate.change_resolution(all_true, raw_bin_size, eval_bin_size)
                pred_for_eval = evaluate.change_resolution(all_pred, raw_bin_size, eval_bin_size)
                performance = evaluate.get_performance(true_for_eval, pred_for_eval, targets, 'whole')
                performance_per_resolution.append([raw_bin_size, eval_bin_size] + list(performance.mean().values))
    metric = 'pr_corr'
    label = 'Pearson\'s r'
    sorted_personr = pd.DataFrame(performance_per_resolution,
                                  columns=['train', 'eval'] + list(performance.columns[:-1].values)).sort_values(
        ['train', 'eval'])[['train', 'eval', metric]]

    padded_values = []
    for train_bin, df in sorted_personr.groupby('train'):
        pr_values = list(df[metric].values)
        add_N = len(bin_sizes) - len(pr_values)
        if add_N > 0:
            pr_values = [np.nan for n in range(add_N)] + pr_values
        padded_values.append(pr_values)
    with open(result_path, 'w') as f:
        f.write(json.dumps(padded_values))
