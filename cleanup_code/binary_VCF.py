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

folder = sys.argv[1]
layer_flag = sys.argv[2]
func_flag = sys.argv[3]

if 'binary' in folder:
    if int(layer_flag) == -1:
        if func_flag == 'effect_size':
            output_dir = ('./datasets/VCF/CAGI_results/output_effect')
        elif func_flag == 'log_ratio':
            output_dir = ('./datasets/VCF/CAGI_results/output_ratio')
    elif int(layer_flag) == -2 :
        output_dir = ('./datasets/VCF/CAGI_no_robust/logit_effect')
elif 'coverage' in folder:
    output_dir = ('./datasets/VCF/CAGI_results/coverage_effect')

if os. path. isdir(output_dir) == False:
    os.system('mkdir '+ output_dir)

#get model that we want to run VCF on
from pathlib import Path
model_paths = list(Path(folder).rglob("*.h5"))

# load in dataset that conatin vcf sites
print('load CAGI dataset')
vcf_data = '/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/CAGI5/CAGI_onehot.h5'
f =  h5py.File(vcf_data, "r")
alt_3k = f['alt'][()]
ref_3k = f['ref'][()]
f.close()
print('finished loading')

#loop through models and run function
for model_path in model_paths:
    print('processing model '+ str(model_path))
    #load model
    model_path = str(model_path)
    model = tf.keras.models.load_model(model_path,compile = True)

    # #run robust vcf on both QTL and control
    # vcf_diff = explain.vcf_binary(ref_3k,alt_3k,model,
    #                             layer = layer_flag,diff_func = func_flag)
    # vcf_diff = np.concatenate(vcf_diff)

    #vcf test without robustness run
    alt,ref = custom_fit.center_crop(alt_3k,ref_3k,2048)
    vcf_diff = explain.vcf_binary_fast(ref,alt,model)
    vcf_diff = np.concatenate(vcf_diff)

    #decide ouput directory and file name
    model_name_list = model_path.split('/')
    #vcf_output_path = './datasets/VCF/CAGI_results/'+model_name_list[2]+'.h5'
    vcf_name = model_name_list[2] + '.h5'
    vcf_output_path = os.path.join(output_dir,vcf_name)
    #write output file
    h5_dataset = h5py.File(vcf_output_path, 'w')
    h5_dataset.create_dataset('vcf_diff', data=vcf_diff)
    h5_dataset.close()

    tf.keras.backend.clear_session()
#f.close()
