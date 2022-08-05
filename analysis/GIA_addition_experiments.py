import glob
import os
import pandas as pd
import sys
sys.path.append('../gopher')
import global_importance, utils

# GIA addition experiments
df = pd.read_csv('inter_results/model_evaluations/binloss_basenji_v2.csv')
run_path_basenji_32 = glob.glob('../trained_models/**/' + os.path.basename(
    df[(df['loss_fn'] == 'poisson') & (df['bin_size'] == 32)]['run_dir'].values[0]), recursive=True)[0]
run_path_residualbind = glob.glob('../trained_models/**/run-20211023_095131-w6okxt01', recursive=True)[0]
binary_logit_model_paths = glob.glob('../trained_models/binary/*/run*/logit.h5')
run_paths = [run_path_basenji_32,
             run_path_residualbind] + binary_logit_model_paths  # quant model and logits of binary models
motif_pairs = ['..TGA.TCA..,..TGA.TCA..', '..TGA.TCA..,..GATAA..', '..TGA.TCA..,..ATAAA..']
result_gia_dir = utils.make_dir('inter_results/GIA_files')

for run_path in run_paths:
    for motif_pair in motif_pairs:
        motif_pair = motif_pair.split(',')
        global_importance.gia_add_motifs(run_path, data_dir='../datasets/quantitative_data/testset/',
                                         cell_lines=['PC-3'],
                                         motif_cluster=motif_pair,
                                         out_dir=os.path.join(result_gia_dir, 'gia_add_motifs/'))

# occlusion GIA experiments

testset, targets = utils.collect_whole_testset('../datasets/quantitative_data/testset/', coords=True)
_, np_X, np_Y = utils.convert_tfr_to_np(testset)
threshold = 2
N, L, C = np_Y.shape
mask = np_Y.reshape((-1, L * C)).max(axis=1) > threshold
X_set = np_X[mask]
occl_results = 'inter_results/GIA_files/occlusion.csv'

run_path = glob.glob('../trained_models/**/run-20211023_095131-w6okxt01', recursive=True)[0]
model, _ = utils.read_model(run_path)
gi = global_importance.GlobalImportance(model, targets)

gi.occlude_all_motif_instances(X_set, ['TGA.TCA'],
                               num_sample=10000)
results = pd.concat(gi.summary_remove_motifs)
results['N instances'] = [int(r) for r in results['N instances'].values]
results.to_csv(occl_results)
