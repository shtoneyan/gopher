import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import subprocess
import os, shutil, h5py,scipy
import util
import custom_fit
import seaborn as sns
import modelzoo
import explain
import tfr_evaluate
import glob
import sys


exp_flag = sys.argv[1]

#get model that we want to run VCF on

if exp_flag == 'augmentation':
    model_paths = []
    all_run_metadata = []
    model_path_pair = {'Basenji 128':'paper_runs/basenji/augmentation_basenji/*','Bpnet 1':'paper_runs/bpnet/augmentation_48/*'}
    for model_bin,dir_path in model_path_pair.items():
        temp_metadata = []
        for run_path in glob.glob(dir_path):
             temp_metadata.append(tfr_evaluate.get_run_metadata(run_path))
        temp_metadata = pd.concat(temp_metadata)
        temp_metadata['dataset'] = ['random_chop' if 'random' in data_dir else 'peak_centered' for data_dir in temp_metadata['data_dir'].values]

        bin_size = int(model_bin.split(' ')[-1])
        for i, df in temp_metadata[temp_metadata['bin_size']==bin_size].groupby(['crop', 'rev_comp', 'dataset']):
            assert df.shape[0] == 3, 'mip'""
            all_run_metadata.append(df.iloc[0])
    all_run_metadata = pd.DataFrame(all_run_metadata)
    model_paths = all_run_metadata['run_dir']
    output_dir = './datasets/VCF/CAGI_no_robust/augmentation/'

if exp_flag == 'bin_size':
    model_paths = []
    all_run_metadata = []
    model_path_pair = {'Basenji':'paper_runs/basenji/binloss_basenji/*','Bpnet':'paper_runs/bpnet/bin_loss_40/*'}
    for model_bin,dir_path in model_path_pair.items():
        for run_path in glob.glob(dir_path):
            all_run_metadata.append(tfr_evaluate.get_run_metadata(run_path))
    all_run_metadata = pd.concat(all_run_metadata)
    all_run_metadata = all_run_metadata[all_run_metadata['loss_fn']=='poisson']
    model_paths = all_run_metadata['run_dir'].values
    output_dir = './datasets/VCF/CAGI_no_robust/bin_size/'

if exp_flag == 'new_model':
    model_paths = []
    all_run_metadata = []
    model_path_pair = {'32 res':'paper_runs/new_models/32_res/*','base res':'paper_runs/new_models/base_res/*'}
    for model_bin,dir_path in model_path_pair.items():
        for run_path in glob.glob(dir_path):
            all_run_metadata.append(tfr_evaluate.get_run_metadata(run_path))
    all_run_metadata = pd.concat(all_run_metadata)
    model_paths=all_run_metadata['run_dir'].values
    output_dir = './datasets/VCF/CAGI_no_robust/new_model/'

if os. path. isdir(output_dir) == False:
    os.system('mkdir '+ output_dir)

#save metadata csv to output dir
csv_output = os.path.join(output_dir,'run_metadata.csv')
all_run_metadata.to_csv(csv_output)

# load in dataset that conatin both dsQTL sites and control SNP sites
vcf_data = '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/CAGI5/CAGI_onehot.h5'
f =  h5py.File(vcf_data, "r")
alt_3k = f['alt'][()]
ref_3k = f['ref'][()]
f.close()

#loop through models and run function
for model_path in model_paths:
    #load model
    model = modelzoo.load_model(model_path,compile = True)

    # #run robust vcf on both QTL and control
    # vcf_diff = explain.vcf_robust(ref_3k,alt_3k,model)
    # run vcf with no robustness test
    alt,ref = custom_fit.center_crop(alt_3k,ref_3k,2048)
    vcf_diff = explain.vcf_fast(ref,alt,model)
    vcf_diff = np.concatenate(vcf_diff)

    #decide ouput directory and file name
    vcf_output_path = os.path.join(output_dir,model_path.split('/')[-1]+'.h5')

    #write output file
    h5_dataset = h5py.File(vcf_output_path, 'w')
    h5_dataset.create_dataset('vcf_diff', data=vcf_diff)
    #h5_dataset.create_dataset('background_diff',data = background_diff)
    h5_dataset.close()

    tf.keras.backend.clear_session()
