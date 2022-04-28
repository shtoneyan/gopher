import sys
sys.path.append('../gopher')
import modelzoo
import pandas as pd
import numpy as np
import glob
import utils
import filter_viz
import subprocess
import os

binary_model = glob.glob('../trained_models/binary/*/*')
bpnet = ['../trained_models/bpnet/augmentation_48/run-20211006_190817-456uzbu4']
basenji_128 = ['../trained_models/basenji_v2/binloss_basenji_v2/run-20220406_162758-bpl8g29s']
new_model = glob.glob('../trained_models/new_models/*/*/*/*/*')
quantitative_model = new_model + basenji_128+bpnet
model_compile = binary_model + quantitative_model

output_dir = './inter_result/tomtom/'
profile_data_dir = '../datasets/training_datasets/peak_centered/i_2048_w_1'
testset = utils.make_dataset(profile_data_dir, 'test', utils.load_stats(profile_data_dir), batch_size=128,shuffle = False)
test_x = testset.map(lambda x,y: x)

# for model in model_compile:
for model in test_CNN:
    if 'basenji' in model:
        if 'exp' in model:
            filter_layer = 4
        else:
            filter_layer = 3
    elif 'basset' in model:
        filter_layer = 2
    elif 'bpnet' in model:
        filter_layer = 1
    elif any(x in model for x in ['new_model','cnn','residual']):
        filter_layer = 3
    else:
        print('error in filter layer assignment')
        print(model)
    tfmodel = utils.read_model(model)[0]
    max_filter,counter = filter_viz.filter_max_align_batch(test_x,tfmodel,layer = filter_layer)
    clip_filter = filter_viz.clip_filters(max_filter, threshold=0.5, pad=3)
    output_pre = output_dir+model.split('/')[-1]
    filter_viz.meme_generate(clip_filter,output_file = output_dir+model.split('/')[-1]+'.txt')

filter_output = glob.glob('./inter_result/tomtom/*')
for model_filter in filter_output:
    run_name = model_filter.split('/')[-1].split('.')[0]
    output_dir = './inter_result/tomtom/'+run_name+'/'
    if os.path.isdir(output_dir):
        print(output_dir+' existed')
        continue
    cmd = 'tomtom -o ./inter_result/tomtom/'+run_name+'/ '+model_filter+' ../datasets/JASPAR2022_CORE_vertebrates.meme'
    subprocess.call(cmd,shell = True)
    mv_cmd = 'mv '+model_filter+' ./inter_result/tomtom/'+run_name+'/'
    subprocess.call(mv_cmd,shell = True)
