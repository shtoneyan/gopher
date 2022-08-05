import sys
sys.path.append("../gopher")
import robustness_test
import h5py
import matplotlib.pyplot as plt
import glob

#Get all basenji_v2 augmentation experiments and run robustness evaluation
model_paths = glob.glob('../trained_models/basenji_v2/augmentation_basenji_v2/*')
testset_path = '../datasets/quantitative_data/peak_centered/i_3072_w_1/'
output_dir = './inter_result/basenji_v2_robust/'
robustness_test.get_robustness_values(model_paths,testset_path,output_dir=output_dir,intermediate = True,batch_size = 32)

#Get all bpnet augmentation experiments and run robustness evaluation
model_paths = glob.glob('../trained_models/bpnet/augmentation_48/128/*')
testset_path = '../datasets/quantitative_data/peak_centered/i_3072_w_1/'
output_dir = './inter_result/bpnet_robust/'
robustness_test.get_robustness_values(model_paths,testset_path,output_dir=output_dir,intermediate = True,batch_size = 32)
