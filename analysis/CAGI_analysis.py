import sys
sys.path.append('../gopher')
import h5py
import glob
import pandas as pd
import utils
import os
import variant_effect

#Across new models and binary models
binary_model = glob.glob('../trained_models/binary/*/*')
new_model = glob.glob('../trained_models/new_models/*/*/*/*/*')
model_compile = new_model + binary_mode

#Read in created dataset
onehot_ref_alt = h5py.File('../dataset/CAGI_onehot.h5', 'r')
ref = onehot_ref_alt['ref'][()]
alt = onehot_ref_alt['alt'][()]

for model in model_compile:
    if 'new_models' in model:
        robust_output_dir = './inter_result/CAGI_results/new_models/'+('/'.join(model.split('/')[4:-1]))
        non_robust_output_dir = './inter_result/CAGI_no_robust/new_models/'+('/'.join(model.split('/')[4:-1]))
    elif 'binary' in model:
        robust_output_dir = './inter_result/CAGI_results/binary/'+('/'.join(model.split('/')[4:-1]))
        non_robust_output_dir = './inter_result/CAGI_no_robust/binary/'+('/'.join(model.split('/')[4:-1]))

    if os.path.isdir(robust_output_dir)==False:
        os.makedirs(robust_output_dir)

    if os.path.isdir(non_robust_output_dir)==False:
        os.makedirs(non_robust_output_dir)

    variant_effect.vcf_quantitative(model,ref,alt,2048,
                                    robust_output_dir+'/'+model.split('/')[-1],
                                    robust = True)

    variant_effect.vcf_quantitative(model,ref,alt,2048,
                                    non_robust_output_dir+'/'+model.split('/')[-1],
                                    robust = False)
#Across augmentation experiment
model_paths = []
all_run_metadata = []
model_path_pair = {'Basenji 128':'../trained_models/basenji_v2/augmentation_basenji_v2/*','Bpnet 1':'../trained_models/bpnet/augmentation_48/1/*'}
for model_bin,dir_path in model_path_pair.items():
    temp_metadata = []
    for run_path in glob.glob(dir_path):
         temp_metadata.append(utils.get_run_metadata(run_path))
    temp_metadata = pd.concat(temp_metadata)
    temp_metadata['dataset'] = ['random_chop' if 'random' in data_dir else 'peak_centered' for data_dir in temp_metadata['data_dir'].values]

    bin_size = int(model_bin.split(' ')[-1])
    for i, df in temp_metadata[temp_metadata['bin_size']==bin_size].groupby(['crop', 'rev_comp', 'dataset']):
        assert df.shape[0] == 3, 'mip'""
        all_run_metadata.append(df.iloc[0])
all_run_metadata = pd.DataFrame(all_run_metadata)
model_paths = all_run_metadata['run_dir']
n_robust_output_dir = './inter_result/CAGI_no_robust/augmentation/'
robust_output_dir = './inter_result/CAGI_results/augmentation/'

csv_output = os.path.join(robust_output_dir,'run_metadata.csv')
all_run_metadata.to_csv(csv_output)
csv_output = os.path.join(n_robust_output_dir,'run_metadata.csv')
all_run_metadata.to_csv(csv_output)

onehot_ref_alt = h5py.File('../datasets/CAGI_onehot.h5', 'r')
ref = onehot_ref_alt['ref'][()]
alt = onehot_ref_alt['alt'][()]

for model in model_paths:
    print(model)
    print(robust_output_dir)
    variant_effect.vcf_quantitative(model,ref,alt,2048,
                                    robust_output_dir+'/'+model.split('/')[-1],
                                    robust = True)
    print(n_robust_output_dir)
    variant_effect.vcf_quantitative(model,ref,alt,2048,
                                    n_robust_output_dir+'/'+model.split('/')[-1],
                                    robust = False)
#Across bin loss experiments
model_paths = []
all_run_metadata = []
model_path_pair = {'Basenji':'../trained_models/basenji_v2/binloss_basenji_v2/*',
                   'Bpnet':'../trained_models/bpnet/bin_loss_40/*'}
for model_bin,dir_path in model_path_pair.items():
    for run_path in glob.glob(dir_path):
        all_run_metadata.append(utils.get_run_metadata(run_path))
all_run_metadata = pd.concat(all_run_metadata)
all_run_metadata = all_run_metadata[all_run_metadata['loss_fn']=='poisson']
model_paths = all_run_metadata['run_dir'].values
n_robust_output_dir = './inter_result/CAGI_no_robust/bin_size/'
robust_output_dir = './inter_result/CAGI_results/bin_size/'

csv_output = os.path.join(robust_output_dir,'run_metadata.csv')
all_run_metadata.to_csv(csv_output)
csv_output = os.path.join(n_robust_output_dir,'run_metadata.csv')
all_run_metadata.to_csv(csv_output)

onehot_ref_alt = h5py.File('../datasets/CAGI_onehot.h5', 'r')
ref = onehot_ref_alt['ref'][()]
alt = onehot_ref_alt['alt'][()]

for model in model_paths:
    variant_effect.vcf_quantitative(model,ref,alt,2048,
                                    robust_output_dir+'/'+model.split('/')[-1],
                                    robust = True)
    variant_effect.vcf_quantitative(model,ref,alt,2048,
                                    n_robust_output_dir+'/'+model.split('/')[-1],
                                    robust = False)
