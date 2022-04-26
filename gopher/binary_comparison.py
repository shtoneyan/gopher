import tensorflow as tf
import utils
import numpy as np
import h5py
import scipy
import modelzoo
import os
import json
import sklearn
import utils

def cov_pearson(run_dir,profile_data_dir):
    """
    Calculates Pearson's R with resolution of model input size
    :param run_dir: Run directory of a quantitative model
    :param profile_data_dir: Data direcotry on which the evaluation is done, need to be a quantiative dataset
    :return: A list consist of a Pearson's R per prediction task
    """
    model = utils.read_model(run_dir,False)[0]
    testset = utils.make_dataset(profile_data_dir, 'test', utils.load_stats(profile_data_dir), batch_size=128)
    json_path = os.path.join(profile_data_dir, 'statistics.json')
    with open(json_path) as json_file:
        params = json.load(json_file)

    target_list = []
    pred_list = []
    for i, (x, y) in enumerate(testset):
        target_cov = np.average(y.numpy(),axis = 1)
        pred_cov = model.predict(x)
        if len(pred_cov.shape) == 3:
            pred_cov = tf.reduce_mean(pred_cov,axis = 1)
        target_list.append(target_cov)
        pred_list.append(pred_cov)

    target = np.concatenate(target_list)
    pred = np.concatenate(pred_list)
    r_list = []
    for i in range(0,target.shape[1]):
        r_list.append(scipy.stats.pearsonr(target[:,i],pred[:,i])[0])

    return r_list

def binary_metrics(run_dir,binary_data_dir):
    """
    This function can take both binary or quantitative model and outputs AUPR and AUROC
    :param run_dir: Run directory of the evaluated model
    :param binary_data_dir: Data direcotry on which the evaluation is done, need to be a binary dataset
    :return: Average AURP and AUROC across all prediction targets
    """
    model = utils.read_model(run_dir,False)[0]
    f = h5py.File(binary_data_dir,'r')
    x_test = f['x_test']
    y_test = f['y_test']


    #y_pred = model.predict(x_test)
    y_pred = utils.predict_np(x_test,model)
    if len(y_pred.shape) == 3:
        cov_pred = np.sum(y_pred,axis = 1)
    else:
        cov_pred = y_pred
    aupr = []
    auroc = []
    for a in range(0,y_test.shape[1]):
        precision,recall,threshold = sklearn.metrics.precision_recall_curve(y_test[:,a],cov_pred[:,a])
        fpr,tpr,threshold = sklearn.metrics.roc_curve(y_test[:,a],cov_pred[:,a])
        aupr.append(sklearn.metrics.auc(recall,precision))
        auroc.append(sklearn.metrics.auc(fpr,tpr))
    f.close()
    return np.mean(aupr),np.mean(auroc)


def binary_to_profile(binary_model_dir,profile_data_dir):
    """
    This function evaluates the performance of a binary model using Pearson's R
    :param binary_model_dir: Save directory for the binary model
    :param profile_data_dir: Data direcotry on which the evaluation is done, need to be a quantiative dataset
    :return: A list consist of a Pearson's R per prediction task
    """
    model =utils.read_model(binary_model_dir,False)[0]
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                  outputs=model.layers[-2].output)

    testset = utils.make_dataset(profile_data_dir, 'test', utils.load_stats(profile_data_dir), batch_size=128)
    json_path = os.path.join(profile_data_dir, 'statistics.json')
    with open(json_path) as json_file:
        params = json.load(json_file)

    target_list = []
    pred_list = []
    for i, (x, y) in enumerate(testset):
        target_cov = np.average(y.numpy(),axis = 1)
        pred_cov = intermediate_layer_model.predict(x)
        target_list.append(target_cov)
        pred_list.append(pred_cov)

    target = np.concatenate(target_list)
    pred = np.concatenate(pred_list)

    target = np.squeeze(target)
    pred = np.squeeze(pred)

    r_list = []
    for i in range(0,target.shape[1]):
        r_list.append(scipy.stats.pearsonr(target[:,i],pred[:,i])[0])

    return r_list
