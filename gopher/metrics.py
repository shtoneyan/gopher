import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import scipy
import sklearn.metrics as skm
from tensorflow.python.keras import backend as K
from scipy.spatial import distance
from scipy import stats


def get_correlation_concatenated(all_truth, all_pred, corr_type='pearsonr'):
    """
    Function to calculate concatenated correlation coefficient per target class
    :param all_truth: ground truth in a numpy array
    :param all_pred: predictions in a numpy array
    :param corr_type: pearsonr or spearmanr
    :return: correlation values
    """
    pr_all = []
    N, L, C = all_truth.shape
    flat_truth = all_truth.reshape(N * L, C)
    flat_pred = all_pred.reshape(N * L, C)
    for c in range(C):
        pr = eval('stats.' + corr_type)(flat_truth[:, c], flat_pred[:, c])[0]
        pr_all.append(pr)
    return np.array(pr_all)


def get_correlation_per_seq(all_truth, all_pred, take_avg=True,
                            corr_type='pearsonr'):
    """
    Function to calculate per sequence correlation coefficients per class
    :param all_truth: ground truth np array
    :param all_pred: prediction np array
    :param take_avg: compute average excluding nan values
    :param corr_type: pearsonr or spearmanr
    :return: per sequence per target correlation
    """
    avg_per_cell_line = []
    N, L, C = all_truth.shape
    for c in range(C):
        pr_values = []
        for n in range(N):
            pr = eval('stats.' + corr_type)(all_truth[n, :, c], all_pred[n, :, c])[0]
            pr_values.append(pr)
        if take_avg:
            avg_per_cell_line.append(np.nanmean(pr_values))
        else:
            avg_per_cell_line.append(pr_values)
    return avg_per_cell_line


def get_mse(a, b):
    """Calculate MSE"""
    return ((a - b) ** 2)


def get_js_per_seq(x, y):
    """
    Function to calculate per sequence JS distance
    :param x: array 1
    :param y: array 2
    :return: JS distance along dim 1
    """
    pseudocount = np.finfo(float).eps
    norm_arrays = []
    for array in [x, y]:
        array = np.clip(array, 0, array.max())
        array += pseudocount
        norm_array = array / np.expand_dims(array.sum(axis=1), 1)
        norm_arrays.append(norm_array)
    return distance.jensenshannon(norm_arrays[0], norm_arrays[1], axis=1)


def get_js_concatenated(x, y):
    """
    Function to calculate concatenated JS distance
    :param x: array 1
    :param y: array 2
    :return: per target concatenated JS distance
    """
    pseudocount = np.finfo(float).eps
    js_conc_per_cell_line = []
    C = x.shape[-1]
    for c in range(C):
        norm_arrays = []
        for raw_array in [x[:, :, c], y[:, :, c]]:
            raw_array = raw_array.flatten()
            array = np.clip(raw_array, 0, raw_array.max())
            array += pseudocount
            norm_array = array / array.sum()
            norm_arrays.append(norm_array)
        js_conc_per_cell_line.append(distance.jensenshannon(norm_arrays[0], norm_arrays[1]))
    return np.array(js_conc_per_cell_line)


def get_poiss_nll(y_true, y_pred):
    """
    Function to calculate poisson NLL
    :param y_true: ground truth np array
    :param y_pred: prediction np array
    :return: poisson NLL
    """
    y_pred_pseudo = y_true + 1
    y_true_pseudo = y_pred + 1
    return y_pred_pseudo - y_true_pseudo * np.log(y_pred_pseudo)


class PearsonR(tf.keras.metrics.Metric):
    def __init__(self, num_targets, summarize=True, name='pearsonr', **kwargs):
        super(PearsonR, self).__init__(name=name, **kwargs)
        self._summarize = summarize
        self._shape = (num_targets,)
        self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

        self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
        self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
        self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
        self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
        self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

    def update_state(self, y_true, y_pred,**kwargs):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0, 1])
        self._product.assign_add(product)

        true_sum = tf.reduce_sum(y_true, axis=[0, 1])
        self._true_sum.assign_add(true_sum)

        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0, 1])
        self._true_sumsq.assign_add(true_sumsq)

        pred_sum = tf.reduce_sum(y_pred, axis=[0, 1])
        self._pred_sum.assign_add(pred_sum)

        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0, 1])
        self._pred_sumsq.assign_add(pred_sumsq)

        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=[0, 1])
        self._count.assign_add(count)

    def result(self):
        true_mean = tf.divide(self._true_sum, self._count)
        true_mean2 = tf.math.square(true_mean)
        pred_mean = tf.divide(self._pred_sum, self._count)
        pred_mean2 = tf.math.square(pred_mean)

        term1 = self._product
        term2 = -tf.multiply(true_mean, self._pred_sum)
        term3 = -tf.multiply(pred_mean, self._true_sum)
        term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
        pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
        tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
        correlation = tf.divide(covariance, tp_var)

        if self._summarize:
            return tf.reduce_mean(correlation)
        else:
            return correlation

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])
