import evaluate
import pandas as pd
import sys
sys.path.append('../gopher')
import utils
import numpy as np
import json


result_base_dir = utils.make_dir('inter_results')
# get datasets
testset, targets = utils.collect_whole_testset('../datasets/quantitative_data/testset/')

model_labels = ['binloss_basenji_v2', 'bpnet_bin_loss_40']
for model_label in model_labels:
    summary_performance = pd.read_csv(result_base_dir+'/model_evaluations/{}.csv'.format(model_label))

    poisson_df = summary_performance[(summary_performance['targets'] == 'PC-3') &
                                     (summary_performance['loss_fn'] == 'poisson')]

    bin_run = dict(zip(poisson_df['bin_size'].values, poisson_df['run_dir'].values))
    # get performance metrics for various evaluation bin sizes
    result_path = result_base_dir+'/{}_triangle_plot.txt'.format(model_label)
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
