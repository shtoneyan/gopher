import tensorflow as tf
import h5py
import explain
import custom_fit
import modelzoo
import os,json
import util
import time
import pandas as pd
import sys
import numpy as np
import glob
import tfr_evaluate, test_to_bw_fast

def main():
    model_name = 'basenji'
    model_paths = glob.glob('paper_runs/{}/augmentation_*/*'.format(model_name))
    # model_paths = glob.glob('paper_runs/basenji/augmentation_basenji/*')
    testset_path = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/4grid_atac/complete/peak_centered/i_3072_w_1/'
    type_id, size_id = testset_path.rstrip('/').split('/')[-2:]
    size_id = size_id.split('_')[1]
    testset_id = '{}_{}_'.format(type_id, size_id)
    eval_type = 'idr'
    save_performance = True
    base_directory = util.make_dir('/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/robustness_test/'+model_name)
    testset, targets, stats = test_to_bw_fast.read_dataset(testset_path, True)

    # overlapping_testset_path = 'datasets/step1K_chr8_whole/i_3072_w_1/'

    for model_path in model_paths:
        new_dir = os.path.join(base_directory, testset_id + model_path.split('/')[-1])
        if not os.path.isdir(new_dir):
            start_time = time.time()
            print(model_path)
            output_directory = util.make_dir(new_dir)
            # bw_filepath_suffix = '_pred.bw'
            variance_dataset_path = os.path.join(output_directory, 'variance_of_preds.h5')


            # load model and datasets
            model = modelzoo.load_model(model_path, compile=True)

            # compute variance of predictions and avg predictions
            predictions_and_variance, center_1K_coordinates, center_ground_truth_1K = explain.batch_pred_robustness_test(testset, stats,
                                                                                                                        model,
                                                                                                                        batch_size=1,
                                                                                                                        shift_num=20,
                                                                                                                        get_preds=save_performance)
            # save variance as h5
            h5_dataset = h5py.File(variance_dataset_path, 'w')
            h5_dataset.create_dataset('prediction_variance', data=np.concatenate(predictions_and_variance['var'], axis=0))
            h5_dataset.create_dataset('center_pred', data=np.concatenate(predictions_and_variance['center_pred'], axis=0))
            h5_dataset.create_dataset('robust_pred', data=np.concatenate(predictions_and_variance['robust_pred'], axis=0))
            h5_dataset.create_dataset('center_ground_truth_1K', data=np.concatenate(center_ground_truth_1K, axis=0))
            h5_dataset.close()

            # if save_performance:
            #     performance_result_path = os.path.join(output_directory, eval_type + '_performance.csv')
            #     # save performance with metadata
            #     performance = tfr_evaluate.get_performance(Y, preds, targets, eval_type)
            #     metadata = tfr_evaluate.get_run_metadata(model_path)
            #     metadata_broadcasted = pd.DataFrame(np.repeat(metadata.values, performance.shape[0], axis=0), columns=metadata.columns)
            #     complete_dataset = pd.concat([performance, metadata_broadcasted], axis=1)
            #     complete_dataset.to_csv(performance_result_path)

            print('OVERALL TIME: '+ str((time.time()-start_time)//60))
        else:
            print('Skipping model '+model_path)

def write_predictions_to_bw(preds, bw_path, cell_line=0,
                            chrom_size_path="/home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv"):
    """create and write bw file with robust predictions"""
    opened_bw = test_to_bw_fast.open_bw(bw_path, chrom_size_path)
    for i in range(preds.shape[0]):
        add_to_bw(preds[i,:,cell_line], coords[i], opened_bw)
    opened_bw.close()

def add_to_bw(value, coord, bw_file_object, step=1):
    '''Write tf coord, values to open bw file'''
    chrom, start, end = coord
    bw_file_object.addEntries(chrom, int(start), values=value, span=step,
                              step=step)


# __main__
################################################################################
if __name__ == '__main__':
    main()
