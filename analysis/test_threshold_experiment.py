import glob
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np
sys.path.append('../gopher')
import utils, metrics

all_loss_run_paths = glob.glob('../trained_models/basenji_v2/binloss_basenji_v2/*')
poisson_runs = []
for run_path in all_loss_run_paths:
    if utils.get_config(run_path)['loss_fn']['value'] == 'poisson':
        poisson_runs.append(run_path)

testset, targets = utils.collect_whole_testset('../datasets/quantitative_data/testset')
np_x, np_y = utils.convert_tfr_to_np(testset)

test_thresh_results = utils.make_dir('inter_results') + '/basenji_v2_test_thresh_all_bins.csv'

test_range = range(25)
pr_dfs = []
for run_path in poisson_runs:
    model, bin_size = utils.read_model(run_path)
    if bin_size < 2000:
        thresh_pr_dict = {}
        thresh_pos_dict = {}

        for thresh in tqdm(test_range):
            bool_mask = np_y.max(axis=1) > thresh  # above threshold in any 1 cell_line
            thresh_inds = np.unique(np.argwhere(bool_mask).flatten())
            filtered_y = np_y[thresh_inds, :, :]
            filtered_x = np_x[thresh_inds, :, :]
            N_seqs = thresh_inds.shape[0]
            binned_y = filtered_y.reshape(N_seqs, 2048 // bin_size, bin_size, len(targets)).mean(axis=2)
            all_preds = utils.predict_np(filtered_x, model, batch_size=32, reshape_to_2D=False)
            thresh_bin_size_pr = np.nanmean(metrics.get_correlation_per_seq(binned_y, all_preds))

            pr_dfs.append(pd.DataFrame([bin_size, thresh, thresh_bin_size_pr]).T)
pr_dfs = pd.concat(pr_dfs)
pr_dfs.columns = ['bin size', 'threshold', 'Pearson\'s r']
pr_dfs['bin size int'] = [str(int(b)) for b in pr_dfs['bin size'].values]

pr_dfs = pr_dfs.reset_index().sort_values('bin size')
pr_dfs['bin size'] = [str(int(b)) for b in pr_dfs['bin size'].values]
pr_dfs.to_csv(test_thresh_results)
