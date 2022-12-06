import h5py
from gopher import utils
import numpy as np
import os
import tensorflow as tf
import csv

def get_center_coordinates(coord, conserve_start, conserve_end):
    """

    :param coord:
    :param conserve_start:
    :param conserve_end:
    :return:
    """
    '''Extract coordinates according to robsutness test procedure'''
    chrom, start, end = str(np.array(coord)).strip('\"b\'').split('_')
    start, end = np.arange(int(start), int(end))[[conserve_start, conserve_end]]
    return (chrom, start, end)





def batch_pred_robustness_test(testset, sts, model,
                               shift_num=20, window_size=2048, get_preds=True):
    """
    This function obtains a dictionary of robust predictions (predictions averaged across multiple shifts), center
    predictions and variance based on robustness test for a given test set, center 1K coordinates and center 1K ground
    truth values. During robustness test we shift a 2K window within a given input 3K sequence multiple times,
    get predictions and calculate the center 1K (overlap region) average predictions as 'robust' prediction and
    variance.
    :param testset: testset in tf dataset format
    :param sts: statistics file of the dataset
    :param model: loaded trained model
    :param shift_num: number of times to shift
    :param window_size: input size
    :param get_preds: bool, save robust and center preds or not
    :return:
    """
    predictions_and_variance = {}
    predictions_and_variance['var'] = []  # prediction variance
    predictions_and_variance['robust_pred'] = []  # robust predictions
    predictions_and_variance['center_pred'] = []  # 1 shot center predictions
    center_1K_coordinates = []  # coordinates of center 1K
    center_ground_truth_1K = []  # center 1K ground truth base res

    chop_size = sts['target_length']
    center_idx = int(0.5 * (chop_size - window_size))
    conserve_size = window_size * 2 - chop_size
    conserve_start = chop_size // 2 - conserve_size // 2
    conserve_end = conserve_start + conserve_size - 1
    flanking_size = conserve_size // 2

    for t, (C, seq, Y) in enumerate(testset):
        batch_n = seq.shape[0]
        shifted_seq,_,shift_idx = utils.window_shift(seq, seq, window_size, shift_num)
        # get prediction for shifted read
        shift_pred = model.predict(shifted_seq)
        bin_size = window_size / shift_pred.shape[1]
        shift_pred = np.repeat(shift_pred, bin_size, axis=1)

        # Select conserve part only
        crop_start_i = conserve_start - shift_idx - center_idx
        crop_idx = crop_start_i[:, None] + np.arange(conserve_size)
        crop_idx = crop_idx.reshape(conserve_size * shift_num * batch_n)
        crop_row_idx = np.repeat(range(0, shift_num * batch_n), conserve_size)
        crop_f_index = np.vstack((crop_row_idx, crop_idx)).T.reshape(shift_num * batch_n, conserve_size, 2)

        # get pred 1k part
        shift_pred_1k = tf.gather_nd(shift_pred[range(shift_pred.shape[0]), :, :], crop_f_index)
        sep_pred = np.array(np.array_split(shift_pred_1k, batch_n))
        var_pred = np.mean(np.std(sep_pred, axis=1),axis = 1)
        cov_pred = np.mean(sep_pred,axis = (2,1))
        predictions_and_variance['var'].append(var_pred/cov_pred)
        if get_preds:
            predictions_and_variance['robust_pred'].append(sep_pred.mean(axis=1))

            # get prediction for center crop
            center_2K_seq = seq[:, conserve_start - flanking_size:conserve_end + 1 + flanking_size, :]
            center_2K_pred = model.predict(center_2K_seq)
            center_2K_pred_unbinned = np.repeat(center_2K_pred, window_size // center_2K_pred.shape[1], axis=1)
            predictions_and_variance['center_pred'].append(
                center_2K_pred_unbinned[:, (conserve_size - flanking_size):(conserve_size + flanking_size), :])

            # add ground truth center 1K coordinates and coverage
            center_1K_coordinates.append([get_center_coordinates(coord, conserve_start, conserve_end) for coord in C])
            center_ground_truth_1K.append(Y.numpy()[:, conserve_start:conserve_end + 1, :])
    return predictions_and_variance, center_1K_coordinates, center_ground_truth_1K


def get_robustness_values(model_paths, testset_path, output_dir='robustness_test_output', intermediate = False, batch_size=1, shift_num=20):
    """
    This function runs robustness test on a given set of models and test set. For each model and for every sequence in
    the test set the following is computed and saved in h5:
    (i) center_ground_truth_1K - ground truth of the center 1K region for which the robustness score is estimated
    (ii) center_pred - one snapshot or single forward pass prediction centered at the center position of the input
    (iii) robust_pred - robust predictions, i.e. averaged predictions of shifted sequences centered after stochastic shifts (by default 20 per sequence)
    (iv) prediction_variance - sum of variation in predictions across the 1K center region (how different the predictions are as we shift the sequence)

    :param model_paths: iterable of model paths
    :param testset_path: path to tfr dataset to test set
    :param output_dir: directory where to save the outputs
    :param batch_size: batch size
    :param shift_num: number of stochastic shifts in each sequence
    :return: None
    """
    stats = utils.load_stats(testset_path)
    utils.make_dir(output_dir)
    testset = utils.make_dataset(testset_path,'test',stats,batch_size = batch_size,shuffle=False)
    robust_dict = {}

    for model_path in model_paths:

        model, _ = utils.read_model(model_path)
        # compute variance of predictions and avg predictions
        predictions_and_variance, center_1K_coordinates, center_ground_truth_1K = batch_pred_robustness_test(testset, stats,
                                                                                                                    model,
                                                                                                                    shift_num=shift_num,
                                                                                                                    get_preds=True)

        if intermediate == True:
            new_dir = os.path.join(output_dir, os.path.basename(os.path.abspath(model_path)))
            if not os.path.isdir(new_dir):
                output_directory = utils.make_dir(new_dir)
            else:
                output_directory = new_dir
            # save variance as h5
            variance_dataset_path = os.path.join(output_directory, model_path.split('/')[-1]+'.h5')
            h5_dataset = h5py.File(variance_dataset_path, 'w')
            h5_dataset.create_dataset('prediction_variance', data=np.concatenate(predictions_and_variance['var'], axis=0))
            h5_dataset.create_dataset('center_pred', data=np.concatenate(predictions_and_variance['center_pred'], axis=0))
            h5_dataset.create_dataset('robust_pred', data=np.concatenate(predictions_and_variance['robust_pred'], axis=0))
            h5_dataset.create_dataset('center_ground_truth_1K', data=np.concatenate(center_ground_truth_1K, axis=0))
            h5_dataset.close()

        mean_var = np.concatenate(predictions_and_variance['var'], axis=0).mean()
        robust_dict[model_path] = mean_var

    variance_dict_path = os.path.join(output_dir, 'average_variance.csv')
    with open(variance_dict_path, 'w') as csvfile:
        writer = csv.writer(csvfile, )
        writer.writerow(['model_path','variation_score'])
        for key,value in robust_dict.items():
            writer.writerow([key,value])
