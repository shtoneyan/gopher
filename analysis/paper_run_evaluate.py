import evaluate
import glob
import os
from tqdm import tqdm

data_dir = '../../datasets/quantitative_data/testset/'
trained_models_dir = '../../trained_models'
output_dir = '../../figure/inter_results/model_evaluations/'

folder_label_pairs = {'basenji_v2/augmentation_basenji_v2': 'basenji_v2_augmentation',
                      'basenji_v2/train_threshold_basenji_v2': 'train_threshold_basenji_v2',
                      'basenji_v2/binloss_basenji_v2': 'binloss_basenji_v2',
                      'bpnet/bin_loss_40': 'bpnet_bin_loss_40',
                      'bpnet/augmentation_48/': 'bpnet_augmentation_48',
                      'new_models': 'new_model'}

scale_these = ['basenji_v2/binloss_basenji_v2', 'bpnet/bin_loss_40']

for folder, label in folder_label_pairs.items():

    evaluate.evaluate_project(data_dir,
                              project_dir=os.path.join(trained_models_dir, folder),
                              output_dir=output_dir,
                              output_prefix=label, fast=True, scale=False)
    if folder in scale_these:
        evaluate.evaluate_project(data_dir,
                                  project_dir=os.path.join(trained_models_dir, folder),
                                  output_dir=output_dir,
                                  output_prefix='scaled_'+label, fast=True, scale=True)

