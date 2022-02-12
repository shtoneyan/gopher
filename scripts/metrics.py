import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import yaml
import h5py
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import scipy
from scipy.fft import fft
import sklearn.metrics as skm
from tensorflow.python.keras import backend as K
import seaborn as sns
from scipy.spatial import distance
from scipy import stats


def get_correlation_concatenated(all_truth, all_pred, corr_type='pearsonr'):
    pr_all = []
    N,L,C = all_truth.shape
    flat_truth = all_truth.reshape(N*L, C)
    flat_pred = all_pred.reshape(N*L, C)
    for c in range(C):
        pr = eval('stats.'+corr_type)(flat_truth[:,c], flat_pred[:,c])[0]
        pr_all.append(pr)
    return np.array(pr_all)

def get_correlation_per_seq(all_truth, all_pred, take_avg=True,
                            corr_type='pearsonr'):
    avg_per_cell_line = []
    N,L,C = all_truth.shape
    for c in range(C):
        pr_values = []
        for n in range(N):
            pr = eval('stats.'+corr_type)(all_truth[n,:,c], all_pred[n,:,c])[0]
            pr_values.append(pr)
        if take_avg:
            avg_per_cell_line.append(np.nanmean(pr_values))
        else:
            avg_per_cell_line.append(pr_values)
    return avg_per_cell_line

def get_mse(a, b):
    return ((a - b)**2)

def get_scaled_mse(all_truth, all_pred):
    N, L, C = all_pred.shape
    flat_pred = all_pred.reshape(N*L, C)
    flat_truth = all_truth.reshape(N*L, C)
    truth_per_cell_line_sum = flat_truth.sum(axis=0)
    pred_per_cell_line_sum = flat_pred.sum(axis=0)
    scaling_factors =  truth_per_cell_line_sum / pred_per_cell_line_sum
    scaled_preds = scaling_factors * flat_pred
    per_seq_scaled_mse = get_mse(flat_truth, scaled_preds)
    return per_seq_scaled_mse.reshape(N, L, C)

def get_js_per_seq(x, y):
    pseudocount = np.finfo(float).eps
    norm_arrays = []
    for array in [x, y]:
        array = np.clip(array,0,array.max())
        array += pseudocount
        norm_array = array/np.expand_dims(array.sum(axis=1), 1)
        norm_arrays.append(norm_array)
    return distance.jensenshannon(norm_arrays[0], norm_arrays[1], axis=1)

def get_js_concatenated(x, y):
    pseudocount = np.finfo(float).eps
    js_conc_per_cell_line = []
    C = x.shape[-1]
    for c in range(C):
        norm_arrays = []
        for raw_array in [x[:,:,c], y[:,:,c]]:
            raw_array = raw_array.flatten()
            array = np.clip(raw_array,0,raw_array.max())
            array += pseudocount
            norm_array = array/array.sum()
            norm_arrays.append(norm_array)
        js_conc_per_cell_line.append(distance.jensenshannon(norm_arrays[0], norm_arrays[1]))
    return np.array(js_conc_per_cell_line)

def get_scipy_pr(y_true, y_pred):

    pr = scipy.stats.pearsonr(y_true, y_pred)[0]
    return pr

def get_scipy_sc(a, b):
    sc = scipy.stats.spearmanr(a, b)
    return sc[0]

def get_poiss_nll(y_true, y_pred):
#     pseudocount = np.finfo(float).eps
    y_pred += 1
    y_true += 1
    return y_pred - y_true * np.log(y_pred)


class PearsonR(tf.keras.metrics.Metric):
  def __init__(self, num_targets,summarize=True, name='pearsonr', **kwargs):
    super(PearsonR, self).__init__(name=name, **kwargs)
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
    self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    self._product.assign_add(product)

    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    self._true_sumsq.assign_add(true_sumsq)

    pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
    self._pred_sum.assign_add(pred_sum)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])
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


def metrify(func):
    '''Wrapper for getting per TF metric from TF losses'''
    def wrapper(y_true,y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
        return func(y_true,y_pred)
    return wrapper


def metric_fftmse(y_true,y_pred):
    '''MSE(FFT) using scipy for not runnign into memory issues during evaluation'''
    return np.mean(np.mean(np.square((scipy.fft.fft(y_true)-scipy.fft.fft(y_pred)).astype('float')), axis=0), axis=0)

def metric_fftabs(y_true,y_pred):
    '''Abs(FFT)'''
    return np.mean((np.abs((scipy.fft.fft(y_true)-scipy.fft.fft(y_pred)).astype('float'), axis=0)))

def metric_pearsonr(y, pred):
    y_true = tf.cast(y, 'float32')
    y_pred = tf.cast(pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])
    true_mean = tf.divide(true_sum, count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(pred_sum, count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = product
    term2 = -tf.multiply(true_mean, pred_sum)
    term3 = -tf.multiply(pred_mean, true_sum)
    term4 = tf.multiply(count, tf.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = true_sumsq - tf.multiply(count, true_mean2)
    pred_var = pred_sumsq - tf.multiply(count, pred_mean2)
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    correlation = tf.divide(covariance, tp_var)
    return correlation

def metric_r2(y, pred):

    y_true = tf.cast(y, 'float32')
    y_pred = tf.cast(pred, 'float32')
    shape = y_true.shape[-1]
    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])

    true_mean = tf.divide(true_sum, count)
    true_mean2 = tf.math.square(true_mean)

    total = true_sumsq - tf.multiply(count, true_mean2)

    resid1 = pred_sumsq
    resid2 = -2*product
    resid3 = true_sumsq
    resid = resid1 + resid2 + resid3

    r2 = tf.ones_like(shape, dtype=tf.float32) - tf.divide(resid, total)
    return r2



def pearsonr_per_seq(y, pred, summarize=False):
    y_true = tf.cast(y, 'float32')
    y_pred = tf.cast(pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[1])

    true_sum = tf.reduce_sum(y_true, axis=[1])
    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[1])
    pred_sum = tf.reduce_sum(y_pred, axis=[1])
    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[1])

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[1])

    true_mean = tf.divide(true_sum, count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(pred_sum, count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = product
    term2 = -tf.multiply(true_mean, pred_sum)
    term3 = -tf.multiply(pred_mean, true_sum)
    term4 = tf.multiply(count, tf.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = true_sumsq - tf.multiply(count, true_mean2)
    pred_var = pred_sumsq - tf.multiply(count, pred_mean2)
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    correlation = tf.divide(covariance, tp_var)
    if summarize:
        nonan_corr = np.nanmean(correlation.numpy(), axis=0)
        # nonan_corr = tf.boolean_mask(correlation, tf.math.is_finite(correlation))
        return nonan_corr
    else:
        return correlation


# def calculate_pearsonr(target,pred):
#     pearson_profile =np.zeros((target.shape[2],len(target)))
#
#     for task_i in range(0,target.shape[2]):
#         for sample_i in range(0,len(target)):
#             pearson_profile[task_i,sample_i]=(pearsonr(target[sample_i][:,task_i],pred[sample_i][:,task_i])[0])
#
#     return pearson_profile


def pearson_volin(pearson_profile,tasks,figsize=(20,5)):
    pd_dict = {}
    for i in range(0,len(tasks)):
        pd_dict[tasks[i]]=pearson_profile[i]

    pearsonr_pd = pd.DataFrame.from_dict(pd_dict)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.violinplot(data=pearsonr_pd)
    return fig


def pearson_box(pearson_profile,tasks,figsize=(20,5)):
    pd_dict = {}
    for i in range(0,len(tasks)):
        pd_dict[tasks[i]]=pearson_profile[i]

    pearsonr_pd = pd.DataFrame.from_dict(pd_dict)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=pearsonr_pd)
    return fig

def permute_array(arr, axis=0):
    """Permute array along a certain axis

    Args:
      arr: numpy array
      axis: axis along which to permute the array
    """
    if axis == 0:
        return np.random.permutation(arr)
    else:
        return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)



def bin_counts_amb(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
    for i in range(outlen):
        iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
        has_amb = np.any(iterval == -1, axis=1)
        has_peak = np.any(iterval == 1, axis=1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout

def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)

def bin_counts_max(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
    return xout


MASK_VALUE = -1
def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)



def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]

def eval_profile(yt, yp,
                 pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 2, 4, 10]):
    """
    Evaluate the profile in terms of auPR

    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold bellow which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # The filtering
    # criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts  of reads
    do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold

    # make sure everything sums to one
    yp = yp / yp.sum(axis=1, keepdims=True)
    fracs = yt / yt.sum(axis=1, keepdims=True)

    yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
    out = []
    for binsize in binsizes:
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambigous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

        imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
        n_positives = np.sum(y_true == 1)
        n_ambigous = np.sum(y_true == -1)
        frac_ambigous = n_ambigous / y_true.size

        # TODO - I used to have bin_counts_max over here instead of bin_counts_sum
        try:
            res = auprc(y_true,
                        np.ravel(bin_counts_max(yp[do_eval], binsize)))
            res_random = auprc(y_true,
                               np.ravel(bin_counts_max(yp_random, binsize)))
        except Exception:
            print('Exception Encountered')
            res = np.nan
            res_random = np.nan

        out.append({"binsize": binsize,
                    "auprc": res,
                    "random_auprc": res_random,
                    "n_positives": n_positives,
                    "frac_ambigous": frac_ambigous,
                    "imbalance": imbalance
                    })

    return pd.DataFrame.from_dict(out)

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

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[0,1])
    self._product.assign_add(product)

    true_sum = tf.reduce_sum(y_true, axis=[0,1])
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=[0,1])
    self._true_sumsq.assign_add(true_sumsq)

    pred_sum = tf.reduce_sum(y_pred, axis=[0,1])
    self._pred_sum.assign_add(pred_sum)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=[0,1])
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=[0,1])
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
