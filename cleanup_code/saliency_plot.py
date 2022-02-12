import numpy as np
from pathlib import Path
import modelzoo
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import metrics
import explain
import util
from loss import *
import os
import json
import logomaker
import sys




#model_path = '/home/shush/profile/QuantPred/datasets/top25/grid2/model_i_1024_w_1_bpnet_mse.h5'
model_path = sys.argv[1]
grid_point = os.path.basename(model_path)
_,_, input_size, _, window, model_name, loss_name = grid_point.split('.h5')[0].split('_')
#data_dir = '/home/shush/profile/QuantPred/datasets/top25/i_1024_w_1'
data_dir = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/4grid_atac/complete/random_chop/i_3072_w_1'
num_task = 15


#load model and test set
def load_basenji(model_path):
    grid_point = os.path.basename(model_path)
    _,_, input_size, _, window, model_name, loss_name = grid_point.split('.h5')[0].split('_')
    loss_type = eval(loss_name)
    custom_layers = {'GELU':modelzoo.GELU,
                   'StochasticReverseComplement':modelzoo.StochasticReverseComplement,
                   'StochasticShift':modelzoo.StochasticShift,
                   'SwitchReverse':modelzoo.SwitchReverse}
    model = tf.keras.models.load_model(model_path,
                               custom_objects=custom_layers,
                               compile=False)
    model.compile(tf.keras.optimizers.Adam(lr=0.001), loss=loss_type)
    return model

#load model, load test set.
model = load_basenji(model_path)
json_path = os.path.join(data_dir, 'statistics.json')
test_data = util.make_dataset(data_dir, 'test', util.load_stats(data_dir),coords=True)
with open(json_path) as json_file:
        params = json.load(json_file)
y_test = util.tfr_to_np(test_data, 'y', (params['test_seqs'], params['seq_length'], params['num_targets']))
x_test = util.tfr_to_np(test_data, 'x', (params['test_seqs'], params['seq_length'], 4))

#select sequences with hight number of reads
top_num = 2
task_top_list = explain.select_top_pred(test_y,num_task,top_num)


fig, axs = plt.subplots(top_num*num_task,1,figsize=(200,15*num_task))
for i in range(0,num_task):
    X = tf.cast(test_x[task_top_list[i]],dtype='float64')
    saliency_map = explain.peak_saliency_map(X,model,class_index = i,window_size = window)
    saliency_map = saliency_map * X

    for n, w in enumerate(saliency_map):
        ax = axs[i*2+n]

        #plot saliency map representation
        saliency_df = pd.DataFrame(w.numpy(), columns = ['A','C','G','T'])
        logomaker.Logo(saliency_df, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])

fig_name = grid_point.split('.h5')[0].split('model_')[1]
plt.savefig('/home/shush/profile/QuantPred/datasets/top25/grid2/saliency_'+fig_name+'.pdf')
