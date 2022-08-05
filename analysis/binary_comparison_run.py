import sys
sys.path.append('../gopher')
import binary_comparison
import pandas as pd
import numpy as np
import glob

# Collect all models that we want to compare
binary_model = glob.glob('../trained_models/binary/*/*')
bpnet = ['../trained_models/bpnet/augmentation_48/run-20211006_190817-456uzbu4/']
basenji_128 = ['../trained_models/basenji_v2/binloss_basenji_v2/run-20220406_162758-bpl8g29s']
new_model = glob.glob('../trained_models/new_models/*/*/*/*/*')
quantitative_model = new_model + basenji_128+bpnet
model_compile = binary_model + quantitative_model

# Binary and Quantitative dataset for both peak centered and whole test chromosome
bi_peak_dataset = '../datasets/binary_data/peak_center_test.h5'
bi_chr_dataset = '../datasets/binary_data/test.h5'
profile_peak_dataset = '../datasets/training_data/peak_centered/i_2048_w_1'
profile_chr_dataset = '../datasets/training_data/random_chop/i_2048_w_1'


#Evaluate using AUPR and AUROC
model_list = []
aupr_list = []
auroc_list = []
p_aupr_list = []
p_auroc_list = []
for model in model_compile:
    aupr,auroc = binary_comparison.binary_metrics(model,bi_chr_dataset)
    p_aupr,p_auroc = binary_comparison.binary_metrics(model,bi_peak_dataset)
    aupr_list.append(aupr)
    auroc_list.append(auroc)
    p_aupr_list.append(p_aupr)
    p_auroc_list.append(p_auroc)
    model_list.append(model)

binary_metric_df = pd.DataFrame({'model':model_list,'chrom aupr':aupr_list,'chrom auroc':auroc_list,
                                'peak aupr':p_aupr_list,'peak auroc':p_auroc_list})

#Evaluate using Pearson's r
pearson_c_list = []
pearson_p_list = []
model_fn = []

for bi_model in binary_model:
    p_r_list = binary_comparison.binary_to_profile(bi_model,profile_peak_dataset)
    c_r_list = binary_comparison.binary_to_profile(bi_model,profile_chr_dataset)
    pearson_c_list.append(np.mean(c_r_list))
    pearson_p_list.append(np.mean(p_r_list))
    model_fn.append(bi_model)

for model in quantitative_model:
    p_r_list = binary_comparison.cov_pearson(model,profile_peak_dataset)
    c_r_list = binary_comparison.cov_pearson(model,profile_chr_dataset)
    pearson_c_list.append(np.mean(c_r_list))
    pearson_p_list.append(np.mean(p_r_list))
    model_fn.append(model)

quantitative_p_df = pd.DataFrame({'model':model_fn,'chr pearson r' :pearson_c_list,'peak pearson r' :pearson_p_list})


performance_df = pd.merge(binary_metric_df,quantitative_p_df)
performance_df = performance_df[['model','chrom auroc','chrom aupr','chr pearson r','peak auroc','peak aupr','peak pearson r']]
performance_df['model type'] = ['Binary' if'binary' in m_f else 'Quantitative' for m_f in performance_df['model']]
performance_df.to_csv('./inter_result/binary_comparison_performance.csv')
