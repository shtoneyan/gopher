import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from util import bin_resolution


class pearsonr_mse(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_mse", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.001 #best
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        mse_loss = mse()
        mse_raw = mse_loss(y_true, y_pred)
        #sum with weight
        total_loss = pr_loss(y_true, y_pred) + self.alpha*mse_raw
        return total_loss

class pearsonr_poisson(tf.keras.losses.Loss):
    def __init__(self, name="pearsonr_poisson", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.1 ###TODO: SET TO 0.001
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        pr_loss = basenjipearsonr()
        pr = pr_loss(y_true, y_pred)
        #poisson part
        poiss_loss = poisson()
        poiss = poiss_loss(y_true, y_pred)
        #sum with weight
        total_loss = (2*pr*poiss)/(pr+poiss)
        return total_loss

class poisson(tf.keras.losses.Loss):
    def __init__(self, name="poisson", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.keras.losses.poisson(y_true, y_pred)

class mse(tf.keras.losses.Loss):
    def __init__(self, name="mse", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.keras.losses.MSE(y_true,y_pred)

class multinomialnll(tf.keras.losses.Loss):
    def __init__(self, name="multinomial", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        logits_perm = tf.transpose(y_pred, (0, 2, 1))
        true_counts_perm = tf.transpose(y_true, (0, 2, 1))
        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        seqlen = tf.cast(tf.shape(y_true)[0],dtype=tf.float32)
        return -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

class multinomialnll_mse(tf.keras.losses.Loss):
    def __init__(self, name="multinomial_mse", **kwargs):
        super().__init__(name=name)
        self.alpha=kwargs.get('alpha')
    def call(self, y_true, y_pred):
        #multinomial part of loss function
        logits_perm = tf.transpose(y_pred[0], (0, 2, 1))
        true_counts_perm = tf.transpose(y_true[0], (0, 2, 1))
        counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)
        # get the sequence length for normalization
        seqlen = tf.cast(tf.shape(y_true[0])[0],dtype=tf.float32)
        mult_loss = -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen

        #MSE part of loss function
        mse_loss = tf.keras.losses.MSE(y_true[1], y_pred[1])

        #sum with weight
        total_loss = mult_loss + self.alpha*mse_loss

        return total_loss

class multinomialnll_mse_reg(tf.keras.losses.Loss):
    def __init__(self, name="multinomialnll_mse_reg", **kwargs):
        super().__init__(name=name)
        self.alpha = kwargs.get('loss_params')
        if not self.alpha:
            print('ALPHA SET TO DEFAULT VALUE!')
            self.alpha = 0.0000001
        # self.alpha=0.001
    def call(self, y_true, y_pred):
        mult_loss = multinomialnll()(y_true, y_pred)

        #MSE part of loss function
        mse_loss = tf.keras.losses.MSE(y_true, y_pred)

        #sum with weight
        total_loss = self.alpha*mult_loss + mse_loss

        return total_loss

class basenjipearsonr (tf.keras.losses.Loss):
    def __init__(self, name="basenjipearsonr", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

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


        return -tf.reduce_mean(correlation)


class r2 (tf.keras.losses.Loss):
    def __init__(self, name="r2", **kwargs):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')
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
        return -tf.reduce_mean(r2)
