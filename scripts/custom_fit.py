import sys, time
import numpy as np
import tensorflow as tf
import utils
import json, os
import metrics
import wandb
import tensorflow_probability as tfp


class RobustTrainer():
    """Custom training loop with flags incoporated"""

    def __init__(self, model, loss, optimizer,
                 input_window, bin_size, num_targets, metrics, ori_bpnet_flag,
                 rev_comp, crop, sigma):
        # Added for data augmentation
        self.window_size = input_window
        self.bin_size = bin_size

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.ori_bpnet_flag = ori_bpnet_flag
        self.rev_comp = rev_comp
        self.crop = crop
        self.sigma = sigma
        self.num_targets = num_targets

        metric_names = []
        for metric in metrics:
            metric_names.append(metric)

        self.metrics = {}
        self.metrics['train'] = MonitorMetrics(metric_names, 'train', self.num_targets)
        self.metrics['val'] = MonitorMetrics(metric_names, 'val', self.num_targets)
        self.metrics['test'] = MonitorMetrics(metric_names, 'test', self.num_targets)

    @tf.function
    # change
    def robust_train_step(self, x, y, metrics):

        # random crop window
        if self.crop == True:
            x, y = random_crop(x, y, self.window_size)
            if self.bin_size > 1:
                y = bin_resolution(y, self.bin_size)
        elif self.crop == False and self.bin_size > 1:
            print('what_is_this')
            y = bin_resolution(y, self.bin_size)

        # reverse complement
        if self.rev_comp:
            x, y = ReverseComplement(x, y)

        with tf.GradientTape() as tape:
            preds = self.model(x, training=True)

            # bpnet original shape adjustment
            if self.ori_bpnet_flag == True:
                true_cov = tf.math.log(tf.math.reduce_sum(y, axis=1) + 1)
                pred_cov = tf.squeeze(preds[1])
                loss = self.loss([y, true_cov], [preds[0], pred_cov])
            else:
                loss = self.loss(y, preds)
        # trace gradient for training
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        if self.ori_bpnet_flag == True:
            metrics.update_running_metrics(y, preds[0])
        else:
            metrics.update_running_metrics(y, preds)
        return loss

    @tf.function
    def test_step(self, x, y, metrics, training=False):
        """test step for a mini-batch and always center crop window"""
        if self.crop == True:
            x, y = center_crop(x, y, int(self.window_size))
            y = bin_resolution(y, self.bin_size)
        elif self.crop == False:
            y = bin_resolution(y, self.bin_size)

        preds = self.model(x, training=training)

        if self.ori_bpnet_flag == True:
            true_cov = tf.math.log(tf.math.reduce_sum(y, axis=1) + 1)
            pred_cov = tf.squeeze(preds[1])
            loss = self.loss([y, true_cov], [preds[0], pred_cov])
            metrics.update_running_metrics(y, preds[0])
        else:
            loss = self.loss(y, preds)
            metrics.update_running_metrics(y, preds)
        return loss

    def robust_train_epoch(self, trainset, num_step,
                           batch_size=128, shuffle=True, verbose=False, store=True):
        """performs a training epoch with attack to inputs"""

        # prepare dataset
        if shuffle:
            trainset.shuffle(buffer_size=batch_size)
        batch_dataset = trainset
        # loop through mini-batches and perform robust training steps
        start_time = time.time()
        running_loss = 0
        for i, (x, y) in enumerate(batch_dataset):
            loss_batch = self.robust_train_step(x, y, self.metrics['train'])
            self.metrics['train'].running_loss.append(loss_batch)
            running_loss += loss_batch
            progress_bar(i + 1, num_step, start_time, bar_length=30, loss=running_loss / (i + 1))

        # store training metrics
        if store:
            if verbose:
                self.metrics['train'].update_print()
            else:
                self.metrics['train'].update()

    def robust_evaluate(self, name, dataset, batch_size=128, verbose=True, training=False):
        """Evaluate model in mini-batches"""
        batch_dataset = dataset
        for i, (x, y) in enumerate(batch_dataset):
            loss_batch = self.test_step(x, y, self.metrics[name])
            self.metrics[name].running_loss.append(loss_batch)

        # store evaluation metrics
        if verbose:
            self.metrics[name].update_print()
        else:
            self.metrics[name].update()

    def predict(self, x, batch_size=128):
        """Get predictions of model"""
        pred = self.model.predict(x, batch_size=batch_size)
        return pred

    def set_early_stopping(self, patience=10, metric='loss', criterion=None):
        """set up early stopping"""
        self.early_stopping = EarlyStopping(patience=patience, metric=metric, criterion=criterion)

    def check_early_stopping(self, name='val'):
        """check status of early stopping"""
        return self.early_stopping.status(self.metrics[name].get(self.early_stopping.metric)[-1])

    def set_lr_decay(self, decay_rate, patience, metric='loss', criterion=None):
        """set up learning rate decay"""
        self.lr_decay = LRDecay(optimizer=self.optimizer, decay_rate=decay_rate,
                                patience=patience, metric=metric, criterion=criterion)

    def check_lr_decay(self, name='loss'):
        """check status and update learning rate decay"""
        self.lr_decay.check(self.metrics['val'].get(self.lr_decay.metric)[-1])

    def get_metrics(self, name, metrics=None):
        """return a dictionary of metrics stored throughout training"""
        if metrics is None:
            metrics = {}
        metrics[name + '_loss'] = self.metrics[name].loss
        for metric_name in self.metrics[name].metric_names:
            metrics[name + '_' + metric_name] = self.metrics[name].get(metric_name)
        return metrics

    def get_current_metrics(self, name, metrics=None):
        """return a dictionary of metrics stored throughout training"""
        if metrics is None:
            metrics = {}

        metrics[name + '_loss'] = self.metrics[name].loss[-1]
        for metric_name in self.metrics[name].metric_names:
            metrics[name + '_' + metric_name] = self.metrics[name].get(metric_name)[-1]
        return metrics

    def set_learning_rate(self, learning_rate):
        """short-cut to set the learning rate"""
        self.optimizer.learning_rate.assign(learning_rate)


# ------------------------------------------------------------------------------------------
# Helper classes
# ------------------------------------------------------------------------------------------
class LRDecay():
    def __init__(self, optimizer, decay_rate=0.3, patience=10, metric='loss', criterion=None):

        self.optimizer = optimizer
        self.lr = optimizer.lr
        self.decay_rate = tf.constant(decay_rate)
        self.patience = patience
        self.metric = metric

        if criterion is None:
            if metric == 'loss':
                criterion = 'min'
            else:
                criterion = 'max'
        self.criterion = criterion
        self.index = 0
        self.initialize()

    def initialize(self):
        if self.criterion == 'min':
            self.best_val = 1e10
            self.sign = 1
        else:
            self.best_val = -1e10
            self.sign = -1

    def status(self, val):
        """check if validation loss is not improving and stop after patience
           runs out"""
        status = False
        if self.sign * val < self.sign * self.best_val:
            self.best_val = val
            self.index = 0
        else:
            self.index += 1
            if self.index == self.patience:
                self.index = 0
                status = True
        return status

    def check(self, val):
        """ check status of learning rate decay"""
        if self.status(val):
            self.decay_learning_rate()
            print('  Decaying learning rate to %.6f' % (self.lr))

    def decay_learning_rate(self):
        """ sets a new learning rate based on decay rate"""
        self.lr = self.lr * self.decay_rate
        self.optimizer.learning_rate.assign(self.lr)


class EarlyStopping():
    def __init__(self, patience=10, metric='loss', criterion=None):

        self.patience = patience
        self.metric = metric

        if criterion is None:
            if metric == 'loss':
                criterion = 'min'
            else:
                criterion = 'max'
        self.criterion = criterion
        self.index = 0
        self.initialize()

    def initialize(self):
        if self.criterion == 'min':
            self.best_val = 1e10
            self.sign = 1
        else:
            self.best_val = -1e10
            self.sign = -1

    def status(self, val):
        """check if validation loss is not improving and stop after patience
           runs out"""
        status = False
        if self.sign * val < self.sign * self.best_val:
            self.best_val = val
            self.index = 0
        else:
            self.index += 1
            if self.index == self.patience:
                self.index = 0
                status = True
        return status


class MonitorMetrics():
    """class to monitor metrics during training"""

    def __init__(self, metric_names, name, num_targets):
        self.name = name
        self.loss = []
        self.running_loss = []

        self.metric_update = {}
        self.metric = {}
        self.metric_names = metric_names
        self.num_targets = num_targets
        self.initialize_metrics(metric_names)

    def initialize_metrics(self, metric_names):
        """metric names can be list or dict"""
        if 'acc' in metric_names:
            self.metric_update['acc'] = tf.keras.metrics.BinaryAccuracy()
            self.metric['acc'] = []
        if 'pearsonr' in metric_names:
            self.metric_update['pearsonr'] = metrics.PearsonR(self.num_targets)
            self.metric['pearsonr'] = []
        if 'auroc' in metric_names:
            self.metric_update['auroc'] = tf.keras.metrics.AUC(curve='ROC')
            self.metric['auroc'] = []
        if 'aupr' in metric_names:
            self.metric_update['aupr'] = tf.keras.metrics.AUC(curve='PR')
            self.metric['aupr'] = []
        if 'cosine' in metric_names:
            self.metric_update['cosine'] = tf.keras.metrics.CosineSimilarity()
            self.metric['cosine'] = []
        if 'kld' in metric_names:
            self.metric_update['kld'] = tf.keras.metrics.KLDivergence()
            self.metric['kld'] = []
        if 'mse' in metric_names:
            self.metric_update['mse'] = tf.keras.metrics.MeanSquaredError()
            self.metric['mse'] = []
        if 'mae' in metric_names:
            self.metric_update['mae'] = tf.keras.metrics.MeanAbsoluteError()
            self.metric['mae'] = []
        if 'poisson' in metric_names:
            self.metric_update['poisson'] = tf.keras.metrics.Poisson()
            self.metric['poisson'] = []

    def update_running_loss(self, running_loss):
        self.running_loss.append(running_loss)
        return np.mean(self.running_loss)

    def update_running_metrics(self, y, preds):
        #  update metric dictionary
        for metric_name in self.metric_names:
            self.metric_update[metric_name].update_state(y, preds)

    def update_running_loss_metric(self, running_loss, y, preds):
        self.update_running_loss(running_loss)
        self.update_running_metrics(y, preds)

    def reset(self):
        for metric_name in self.metric_names:
            self.metric_update[metric_name].reset_states()

    def update(self):
        self.loss.append(np.mean(self.running_loss))
        self.running_loss = []
        for metric_name in self.metric_names:
            self.metric[metric_name].append(np.mean(self.metric_update[metric_name].result()))
        self.reset()

    def update_print(self):
        self.update()
        self.
        print()

    def print(self):
        if self.loss:
            print('  %s loss:   %.4f' % (self.name, self.loss[-1]))
        for metric_name in self.metric_names:
            print("  " + self.name + " " + metric_name + ":\t{:.5f}".format(self.metric[metric_name][-1]))

    def get(self, name):
        if name == 'loss':
            return self.loss
        else:
            return self.metric[name]


# ------------------------------------------------------------------------------
# Useful functions
# ------------------------------------------------------------------------------
def ReverseComplement(seq_1hot, label_profile, chance=0.5):
    """

    :param seq_1hot:
    :param label_profile:
    :param chance:
    :return:
    """
    rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
    rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
    rc_profile = tf.reverse(label_profile, axis=[1])
    reverse_bool = tf.random.uniform(shape=[]) > (1 - chance)
    src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
    src_profile = tf.cond(reverse_bool, lambda: rc_profile, lambda: label_profile)
    return src_seq_1hot, src_profile


def random_crop(x, y, window_size):
    """

    :param x:
    :param y:
    :param window_size:
    :return:
    """
    # cropping return x_crop and y_crop
    x_dim = x.shape
    if x_dim[1] > window_size:
        indice = (np.arange(window_size) +
                  np.random.randint(low=0, high=x_dim[1] - window_size, size=x_dim[0])[:, np.newaxis])
        indice = indice.reshape(window_size * x_dim[0])
        row_indice = np.repeat(range(0, x_dim[0]), window_size)
        f_index = np.vstack((row_indice, indice)).T.reshape(x_dim[0], window_size, 2)
        x_crop = tf.gather_nd(x, f_index)
        y_crop = tf.gather_nd(y, f_index)
    elif x_dim[1] == window_size:
        x_crop = x
        y_crop = y
    else:
        raise Exception('bad combination of input size and window_size!')

    return x_crop, y_crop


def center_crop(x, y, window_size):
    """

    :param x:
    :param y:
    :param window_size:
    :return:
    """
    x_dim = x.shape
    if len(x_dim) == 2:
        x = np.expand_dims(x, axis=0)
        x_dim = x.shape
    if x_dim[1] > window_size:
        indice = (np.arange(window_size) +
                  np.repeat(int(0.5 * (x_dim[1] - window_size)), x_dim[0])[:, np.newaxis])
        indice = indice.reshape(window_size * x_dim[0])
        row_indice = np.repeat(range(0, x_dim[0]), window_size)
        f_index = np.vstack((row_indice, indice)).T.reshape(x_dim[0], window_size, 2)
        x_crop = tf.gather_nd(x, f_index)
        y_crop = tf.gather_nd(y, f_index)
    elif x_dim[1] == window_size:
        x_crop = x
        y_crop = y

    return x_crop, y_crop


def bin_resolution(y, bin_size):
    """

    :param y:
    :param bin_size:
    :return:
    """
    y_dim = y.shape
    y_bin = tf.math.reduce_mean(tf.reshape(y, (y_dim[0], int(y_dim[1] / bin_size), bin_size, y_dim[2])), axis=2)
    return y_bin


def progress_bar(iter, num_batches, start_time, bar_length=30, **kwargs):
    """plots a progress bar to show remaining time for a full epoch.
       (inspired by keras)"""

    # calculate progress bar
    percent = iter / num_batches
    progress = '=' * int(round(percent * bar_length))
    spaces = ' ' * int(bar_length - round(percent * bar_length))

    # setup text to output
    if iter == num_batches:  # if last batch, then output total elapsed time
        output_text = "\r[%s] %.1f%% -- elapsed time=%.1fs"
        elapsed_time = time.time() - start_time
        output_vals = [progress + spaces, percent * 100, elapsed_time]
    else:
        output_text = "\r[%s] %.1f%%  -- remaining time=%.1fs"
        remaining_time = (time.time() - start_time) * (num_batches - (iter + 1)) / (iter + 1)
        output_vals = [progress + spaces, percent * 100, remaining_time]

    # add performance metrics if included in kwargs
    if 'loss' in kwargs:
        output_text += " -- loss=%.5f"
        output_vals.append(kwargs['loss'])
    if 'acc' in kwargs:
        output_text += " -- acc=%.5f"
        output_vals.append(kwargs['acc'])
    if 'auroc' in kwargs:
        output_text += " -- auroc=%.5f"
        output_vals.append(kwargs['auroc'])
    if 'aupr' in kwargs:
        output_text += " -- aupr=%.5f"
        output_vals.append(kwargs['aupr'])
    if 'pearsonr' in kwargs:
        output_text += " -- pearsonr=%.5f"
        output_vals.append(kwargs['pearsonr'])
    if 'mcc' in kwargs:
        output_text += " -- mcc=%.5f"
        output_vals.append(kwargs['mcc'])
    if 'mse' in kwargs:
        output_text += " -- mse=%.5f"
        output_vals.append(kwargs['mse'])
    if 'mae' in kwargs:
        output_text += " -- mae=%.5f"
        output_vals.append(kwargs['mae'])

    # set new line when finished
    if iter == num_batches:
        output_text += "\n"

    # output stats
    sys.stdout.write(output_text % tuple(output_vals))
