import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os, yaml
# from loss import *
import metrics
from tensorflow import keras


def basenji_v2(input_shape, output_shape, wandb_config={}):
    config = {'filtN_1': 128, 'filtN_2': 1024}
    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]
    window_size = int(input_shape[0]//output_shape[0]//2)
    print(window_size)
    if window_size == 0:
        pool_1 = 1
        window_size = 1
    else:
        pool_1 = 2
    sequence = tf.keras.Input(shape=input_shape, name='sequence')

    current = conv_block(sequence, filters=config['filtN_1'], kernel_size=15, activation='gelu', activation_end=None,
                         strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
                         pool_size=pool_1, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
                         kernel_initializer='he_normal', padding='same')

    current = dilated_residual(current, filters=int(config['filtN_1']//2), kernel_size=3, rate_mult=1.5,
                               conv_type='standard', dropout=0.3, repeat=11, round=False,
                               activation='gelu', batch_norm=True, bn_momentum=0.9)

    current = conv_block(current, filters=config['filtN_2'], kernel_size=1, activation='gelu',
                         dropout=0.05, batch_norm=True, bn_momentum=0.9)

    current = tf.keras.layers.AveragePooling1D(pool_size=window_size)(current)

    outputs = dense_layer(current, output_shape[-1], activation='softplus',
                          batch_norm=False, bn_momentum=0.9)

    model = tf.keras.Model(inputs=sequence, outputs=outputs)
    return model


def basenjimod(input_shape, output_shape, wandb_config={}):
    """
    Basenji-based model that can change the output shape based on bin resolution. Defaults correspond to finetuned values.
    :param input_shape: tuple of input size, e.g. (L, A) = (2048, 4)
    :param output_shape: tuple of output shape, e.g. (L//bin_size, number of targets) = (64, 15)
    :param wandb_config: dictionary of filter sizes, and additional parameters
    :return: model if the filter numbers don't shrink
    """
    config = {'filtN_1': 128, 'filtN_2': 256, 'filtN_4': 256, 'filtN_5': 512,
              'filt_mlt': 1.125, 'add_dropout': False}
    print('Using set of filter sizes for hyperparameter search')
    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}

    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]

    filtN_list = [config[f] for f in ['filtN_1', 'filtN_2', 'filtN_4', 'filtN_5']]
    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
    if config['add_dropout']:
        drp1, drp2, drp4, drp5 = [filt_drp_dict[f] for f in filtN_list]
    else:
        drp1 = drp2 = drp3 = drp4 = drp5 = 0

    # dict for choosing number of maxpools based on output shape
    layer_dict = {32: [1, False],  # if bin size 32 add 1 maxpool and no maxpool of size 2
                  64: [1, True],  # if bin size 64 add 1 maxpool and 1 maxpool of size 2
                  128: [2, False],  # if bin size 128 add 2 maxpool and no maxpool of size 2
                  256: [2, True],  # if bin size 256 add 2 maxpool and 1 maxpool of size 2
                  512: [3, False],
                  1024: [3, True],
                  2048: [4, False]}
    L, _ = input_shape
    n_bins, n_exp = output_shape
    l_bin = L // n_bins
    n_conv_tower, add_2max = layer_dict[l_bin]
    #     n_conv_tower = np.log2(32)
    # print(l_bin, n_conv_tower, add_2max)
    sequence = tf.keras.Input(shape=input_shape, name='sequence')

    current = conv_block(sequence, filters=config['filtN_1'], kernel_size=15, activation='gelu', activation_end=None,
                         strides=1, dilation_rate=1, l2_scale=0, dropout=drp1, conv_type='standard', residual=False,
                         pool_size=8, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
                         kernel_initializer='he_normal', padding='same')

    current, rep_filters = conv_tower(current, filters_init=config['filtN_2'], filters_mult=config['filt_mlt'],
                                      repeat=n_conv_tower,
                                      pool_size=4, kernel_size=5, dropout=drp2, batch_norm=True, bn_momentum=0.9,
                                      activation='gelu')

    if add_2max:
        filtN_3 = int(np.round(rep_filters * config['filt_mlt']))
        if config['add_dropout']:
            drp3 = filt_drp_dict[filtN_3]
        current = conv_block(current, filters=filtN_3, kernel_size=5, activation='gelu', activation_end=None,
                             # changed filter size 5
                             strides=1, dilation_rate=1, l2_scale=0, dropout=drp3, conv_type='standard', residual=False,
                             pool_size=2, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
                             kernel_initializer='he_normal', padding='same')

    current = dilated_residual(current, filters=config['filtN_4'], kernel_size=3, rate_mult=2,
                               conv_type='standard', dropout=drp4, repeat=2, round=False,
                               # repeat=4 TODO:figure out scaling factor for the number of repeats
                               activation='gelu', batch_norm=True, bn_momentum=0.9)

    current = conv_block(current, filters=config['filtN_5'], kernel_size=1, activation='gelu',
                         dropout=drp5, batch_norm=True, bn_momentum=0.9)

    outputs = dense_layer(current, n_exp, activation='softplus',
                          batch_norm=True, bn_momentum=0.9)

    model = tf.keras.Model(inputs=sequence, outputs=outputs)

    if not all(i <= j for i, j in zip(filtN_list, filtN_list[1:])):
        return False
    else:
        return model

def basenji_w1_b64(input_shape, output_shape, wandb_config={}):
    """
    Base resolution basenji-based model. Defaults correspond to finetuned values.
    :param input_shape: tuple of input size, e.g. (L, A) = (2048, 4)
    :param output_shape: tuple of output shape, e.g. (L//bin_size, number of targets) = (64, 15)
    :param wandb_config: dictionary of filter sizes, and additional parameters
    :return: model
    """
    config = {'filtN_1': 128, 'filtN_2': 256, 'filtN_3': 256, 'filtN_4': 256,
              'filtN_5': 256, 'add_dropout': False, 'mult_rate1': 1.125,
              'mult_rate2': 1.125}

    def mult_filt(n, factor=config['mult_rate2']):
        return int(np.round(n * factor))

    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}

    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]

    filtN_list = [config[f] for f in ['filtN_1', 'filtN_2', 'filtN_3', 'filtN_4', 'filtN_5']]
    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
    if config['add_dropout']:
        drp1, drp2, drp3, drp4, drp5 = [filt_drp_dict[f] for f in filtN_list]
    else:
        drp1 = drp2 = drp3 = drp4 = drp5 = 0
    n_exp = output_shape[-1]
    sequence = tf.keras.Input(shape=input_shape, name='sequence')
    current = tf.expand_dims(sequence, -2)

    current = conv_block(current, filters=config['filtN_1'], kernel_size=15, activation='gelu', activation_end=None,
                         strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
                         pool_size=8, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
                         kernel_initializer='he_normal', padding='same', w1=True)

    current, _ = conv_tower(current, filters_init=config['filtN_2'], filters_mult=1.125, repeat=1,
                            pool_size=4, kernel_size=5, batch_norm=True, bn_momentum=0.9,
                            activation='gelu', w1=True)

    current = dilated_residual(current, filters=config['filtN_3'], kernel_size=3, rate_mult=config['mult_rate1'],
                               conv_type='standard', dropout=0.25, repeat=2, round=False,  # repeat=4
                               activation='gelu', batch_norm=True, bn_momentum=0.9, w1=True)

    current = conv_block(current, filters=config['filtN_4'], kernel_size=1, activation='gelu',
                         dropout=0.05, batch_norm=True, bn_momentum=0.9, w1=True)

    # n_filters = 64
    for n in range(2):
        n_filters = mult_filt(config['filtN_4'])
        print(n_filters)
        current = tf.keras.layers.Conv2DTranspose(
            filters=n_filters, kernel_size=(5, 1), strides=(2, 1), padding='same')(current)
        current = tf.keras.layers.UpSampling2D(size=(2, 1))(current)

    current = tf.keras.layers.Conv2DTranspose(
        filters=mult_filt(n_filters), kernel_size=(5, 1), strides=(2, 1), padding='same')(current)
    current = tf.keras.layers.Conv2D(mult_filt(n_filters), 1)(current)
    current = dense_layer(current, n_exp, activation='softplus',
                          batch_norm=True, bn_momentum=0.9)
    outputs = tf.squeeze(
        current, axis=2)
    # upsample

    model = tf.keras.Model(inputs=sequence, outputs=outputs)
    # print(model.summary())
    if not all(i <= j for i, j in zip(filtN_list, filtN_list[1:])):
        return False
    else:
        return model

def basenji_binary(input_shape,exp_num,wandb_config={}):
    config = {'filtN_1': 128, 'filtN_2': 256, 'filtN_4': 256, 'filtN_5': 512,
              'filt_mlt': 1.125, 'add_dropout': False,'activation' :'gelu'}
    print('Using set of filter sizes for hyperparameter search')
    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]

    filtN_list = [config[f] for f in ['filtN_1', 'filtN_2', 'filtN_4', 'filtN_5']]
    filt_drp_dict = {64: 0.1, 128: 0.2, 256: 0.3, 512: 0.4, 1024: 0.5}
    if config['add_dropout']:
        drp1, drp2, drp4, drp5 = [filt_drp_dict[f] for f in filtN_list]
    else:
        drp1 = drp2 = drp3 = drp4 = drp5 = 0

    # dict for choosing number of maxpools based on output shape
    layer_dict = {32: [1, False],  # if bin size 32 add 1 maxpool and no maxpool of size 2
                  64: [1, True],  # if bin size 64 add 1 maxpool and 1 maxpool of size 2
                  128: [2, False],  # if bin size 128 add 2 maxpool and no maxpool of size 2
                  256: [2, True],  # if bin size 256 add 2 maxpool and 1 maxpool of size 2
                  512: [3, False],
                  1024: [3, True],
                  2048: [4, False]}
    L, _ = input_shape
    n_bins, n_exp = (1,exp_num)
    l_bin = L // n_bins
    n_conv_tower, add_2max = layer_dict[l_bin]
    #     n_conv_tower = np.log2(32)
    # print(l_bin, n_conv_tower, add_2max)
    sequence = tf.keras.Input(shape=input_shape, name='sequence')

    current = conv_block(sequence, filters=config['filtN_1'], kernel_size=15, activation=config['activation'], activation_end=None,
                         strides=1, dilation_rate=1, l2_scale=0, dropout=drp1, conv_type='standard', residual=False,
                         pool_size=8, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
                         kernel_initializer='he_normal', padding='same')

    current, rep_filters = conv_tower(current, filters_init=config['filtN_2'], filters_mult=config['filt_mlt'],
                                      repeat=n_conv_tower,
                                      pool_size=4, kernel_size=5, dropout=drp2, batch_norm=True, bn_momentum=0.9,
                                      activation='gelu')

    if add_2max:
        filtN_3 = int(np.round(rep_filters * config['filt_mlt']))
        if config['add_dropout']:
            drp3 = filt_drp_dict[filtN_3]
        current = conv_block(current, filters=filtN_3, kernel_size=5, activation='gelu', activation_end=None,
                             # changed filter size 5
                             strides=1, dilation_rate=1, l2_scale=0, dropout=drp3, conv_type='standard', residual=False,
                             pool_size=2, batch_norm=True, bn_momentum=0.9, bn_gamma=None, bn_type='standard',
                             kernel_initializer='he_normal', padding='same')

    current = dilated_residual(current, filters=config['filtN_4'], kernel_size=3, rate_mult=2,
                               conv_type='standard', dropout=drp4, repeat=2, round=False,
                               # repeat=4 TODO:figure out scaling factor for the number of repeats
                               activation='gelu', batch_norm=True, bn_momentum=0.9)

    current = conv_block(current, filters=config['filtN_5'], kernel_size=1, activation='gelu',
                         dropout=drp5, batch_norm=True, bn_momentum=0.9)

    current = dense_layer(current, n_exp, activation='sigmoid',
                          batch_norm=True, bn_momentum=0.9)
    outputs = tf.keras.layers.Reshape((exp_num,))(current)
    model = tf.keras.Model(inputs=sequence, outputs=outputs)

    #complie with optimizer
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  #metrics = ['mse'])
                  metrics=['binary_accuracy', auroc, aupr])
    model.summary()

    if not all(i <= j for i, j in zip(filtN_list, filtN_list[1:])):
        return False
    else:
        return model

def Basset(inputs,exp_num,padding='same',wandb_config={}):

    config = {'filtN': [300,200,200], 'kern': [10,11,7],
                'dense':1000,'drop_out':[0.3,0.3],'max_pool':[3,4,4],'activation':'relu'}


    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]

    model = tf.keras.models.Sequential([
    #1st conv layer
    tf.keras.layers.Conv1D(config['filtN'][0],config['kern'][0],input_shape = (inputs),
                          kernel_initializer = 'glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation(config['activation']),
    tf.keras.layers.MaxPool1D(pool_size=config['max_pool'][0]),
    #2nd conv layer
    tf.keras.layers.Conv1D(config['filtN'][1],config['kern'][1],
                          kernel_initializer = 'glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool1D(pool_size=config['max_pool'][1]),
    #3rd conv layer
    tf.keras.layers.Conv1D(config['filtN'][2],config['kern'][2],
                          kernel_initializer = 'glorot_normal',
                          padding=padding),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool1D(pool_size=config['max_pool'][2]),

    tf.keras.layers.Flatten(),

    #Fully connected NN
    tf.keras.layers.Dense(config['dense'],activation = 'relu',
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.Dropout(config['drop_out'][0]),
    #2nd layer
    tf.keras.layers.Dense(config['dense'],activation = 'relu',
                          kernel_initializer='glorot_normal'),
    tf.keras.layers.Dropout(config['drop_out'][1]),
    #Sigmoid
    tf.keras.layers.Dense(exp_num,activation='linear',kernel_initializer='glorot_normal'),
    tf.keras.layers.Activation('sigmoid')
    ])

    #complie with optimizer
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  #metrics = ['mse'])
                  metrics=['binary_accuracy', auroc, aupr])
    model.summary()
    return model

def bpnet(input_shape, output_shape, wandb_config={}, softplus=False):
    """

    :param input_shape:
    :param output_shape:
    :param wandb_config:
    :param softplus:
    :return:
    """

    # filtN_1 [64, 128,256]
    # trnaspose kernel_size [7, 17, 25]
    config = {'strand_num': 1, 'filtN_1': 256, 'kern_1': 25,
              'kern_2': 3, 'kern_3': 7, 'layer_num': 10}
    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]

    window_size = int(input_shape[0] / output_shape[0])
    # body
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(config['filtN_1'], kernel_size=config['kern_1'],
                            padding='same', activation='relu')(input)
    for i in range(1, config['layer_num']):
        conv_x = keras.layers.Conv1D(config['filtN_1'],
                                     kernel_size=config['kern_2'],
                                     padding='same', activation='relu',
                                     dilation_rate=2 ** i)(x)
        x = keras.layers.Add()([conv_x, x])

    bottleneck = x

    # heads
    outputs = []
    for task in range(0, output_shape[1]):
        px = keras.layers.Reshape((-1, 1, config['filtN_1']))(bottleneck)
        px = keras.layers.Conv2DTranspose(config['strand_num'],
                                          kernel_size=(config['kern_3'], 1),
                                          padding='same')(px)
        px = keras.layers.Reshape((-1, config['strand_num']))(px)
        px = keras.layers.AveragePooling1D(pool_size=window_size, strides=None,
                                           padding='valid')(px)
        outputs.append(px)

    outputs = tf.keras.layers.concatenate(outputs)

    # TEMP for poisson
    if softplus == True:
        outputs = tf.keras.layers.Activation('softplus')(outputs)
    # outputs = keras.layers.Reshape((output_shape))(outputs)
    model = keras.models.Model([input], outputs)
    return model

def ori_bpnet(input_shape, output_shape, wandb_config={}):
    """

    :param input_shape:
    :param output_shape:
    :param wandb_config:
    :return:
    """
    config = {'strand_num': 1, 'filtN_1': 64, 'kern_1': 25,
              'kern_2': 3, 'kern_3': 25, 'layer_num': 10}
    for k in config.keys():
        if k in wandb_config.keys():
            config[k] = wandb_config[k]
    # body
    window_size = int(input_shape[0] / output_shape[0])
    input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(config['filtN_1'], kernel_size=config['kern_1'],
                            padding='same', activation='relu')(input)
    for i in range(1, config['layer_num']):
        conv_x = keras.layers.Conv1D(config['filtN_1'],
                                     kernel_size=config['kern_2'],
                                     padding='same', activation='relu',
                                     dilation_rate=2 ** i)(x)
        x = keras.layers.Add()([conv_x, x])

    bottleneck = x

    # heads
    profile_outputs = []
    count_outputs = []
    for task in range(0, output_shape[1]):
        # profile shape head
        px = keras.layers.Reshape((-1, 1, config['filtN_1']))(bottleneck)
        px = keras.layers.Conv2DTranspose(config['strand_num'],
                                          kernel_size=(config['kern_3'], 1),
                                          padding='same')(px)
        px = keras.layers.Reshape((-1, config['strand_num']))(px)
        px = keras.layers.AveragePooling1D(pool_size=window_size, strides=None,
                                           padding='valid')(px)
        profile_outputs.append(px)
        # total counts head
        cx = keras.layers.GlobalAvgPool1D()(bottleneck)
        count_outputs.append(keras.layers.Dense(config['strand_num'])(cx))

    profile_outputs = tf.keras.layers.concatenate(profile_outputs)

    count_outputs = tf.keras.layers.concatenate(count_outputs)
    model = keras.models.Model([input], [profile_outputs, count_outputs])
    return model

def conv_binary(input_shape, exp_num, bottleneck=8, wandb_config={}):
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = (1,exp_num)
    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = keras.layers.MaxPool1D(pool_size=8)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    nn = keras.layers.Dense(output_len * bottleneck)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = keras.layers.Dense(num_tasks, activation='sigmoid')(nn)
    outputs = keras.layers.Reshape((num_tasks,))(nn)
    model =  keras.Model(inputs=inputs, outputs=outputs)
    #complie with optimizer
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  #metrics = ['mse'])
                  metrics=['binary_accuracy', auroc, aupr])
    model.summary()
    return model


def residual_binary(input_shape, exp_num, bottleneck=8, wandb_config={}):
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = (1,exp_num)

    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=4)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(512)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    nn = keras.layers.Dense(output_len * bottleneck)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.1)(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)

    nn = keras.layers.Dense(num_tasks, activation='sigmoid')(nn)
    outputs = keras.layers.Reshape((num_tasks,))(nn)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #complie with optimizer
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  #metrics = ['mse'])
                  metrics=['binary_accuracy', auroc, aupr])
    model.summary()
    return model

def conv_profile_task_base(input_shape, output_shape, bottleneck=8, wandb_config={}):
    """
    Task-specific convolutional model that can adapt to various bin sizes
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param bottleneck: bottleneck size
    :param wandb_config: dictionary of parameters including activation function
    :return: model
    """
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = output_shape
    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = keras.layers.MaxPool1D(pool_size=8)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    nn = keras.layers.Dense(output_len * bottleneck)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn_cat = []
    for i in range(num_tasks):
        nn2 = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(nn)
        nn2 = keras.layers.BatchNormalization()(nn2)
        nn2 = keras.layers.Activation('relu')(nn2)
        nn2 = keras.layers.Dropout(0.1)(nn2)
        nn2 = keras.layers.Dense(1, activation='softplus')(nn2)
        nn_cat.append(nn2)
    outputs = tf.concat(nn_cat, axis=2)

    return keras.Model(inputs=inputs, outputs=outputs)

def conv_profile_all_base(input_shape, output_shape, bottleneck=8, wandb_config={}):
    """
    Task-specific convolutional model that can adapt to various bin sizes
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param bottleneck: bottleneck size
    :param wandb_config: dictionary of parameters including activation function
    :return: model
    """
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = output_shape
    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = keras.layers.MaxPool1D(pool_size=8)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    nn = keras.layers.Dense(output_len * bottleneck)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    outputs = keras.layers.Dense(num_tasks, activation='softplus')(nn)
    return keras.Model(inputs=inputs, outputs=outputs)

def residual_profile_task_base(input_shape, output_shape, bottleneck=8, wandb_config={}):
    """
    residual model with task specific heads at base resolution
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param bottleneck: bottleneck size
    :param wandb_config: dictionary of parameters including activation function
    :return: model
    """
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = output_shape

    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=8)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=3)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(512)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    nn = keras.layers.Dense(output_len * bottleneck)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=6)

    nn_cat = []
    for i in range(num_tasks):
        nn2 = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(nn)
        nn2 = keras.layers.BatchNormalization()(nn2)
        nn2 = keras.layers.Activation('relu')(nn2)
        nn2 = keras.layers.Dropout(0.1)(nn2)
        nn2 = keras.layers.Dense(1, activation='softplus')(nn2)
        nn_cat.append(nn2)
    outputs = tf.concat(nn_cat, axis=2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def residual_profile_all_base(input_shape, output_shape, bottleneck=8, wandb_config={}):
    """
    Residual base resolution model without task specific heads
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param bottleneck: bottleneck size
    :param wandb_config: dictionary of parameters including activation function
    :return: model
    """
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = output_shape

    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=192, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=4)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(512)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    nn = keras.layers.Dense(output_len * bottleneck)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Reshape([output_len, bottleneck])(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.1)(nn)
    nn = residual_block(nn, 3, activation='relu', num_layers=5)

    outputs = keras.layers.Dense(num_tasks, activation='softplus')(nn)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def residual_profile_all_dense_32(input_shape, output_shape, wandb_config={}):
    """
    Residual model with no task specific heads at 32 bin resolution
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param wandb_config: dictionary of parameters including activation function
    :return: model
    """
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = output_shape

    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=256, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Conv1D(filters=1024, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=3)
    nn = keras.layers.MaxPool1D(pool_size=2)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    outputs = keras.layers.Dense(num_tasks, activation='softplus')(nn)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def residual_profile_task_conv_32(input_shape, output_shape, wandb_config={}):
    """
    Residual task-specific 32 resolution model
    :param input_shape: tuple of input shape
    :param output_shape: tuple of output shape
    :param wandb_config: dictionary of parameters including activation function
    :return: model
    """
    assert 'activation' in wandb_config.keys(), 'ERROR: no activation defined!'
    output_len, num_tasks = output_shape

    inputs = keras.Input(shape=input_shape, name='sequence')

    nn = keras.layers.Conv1D(filters=256, kernel_size=19, padding='same')(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(wandb_config['activation'], name='filter_activation')(nn)
    nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Conv1D(filters=512, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=5)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Conv1D(filters=1024, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=3)
    nn = keras.layers.MaxPool1D(pool_size=2)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Conv1D(filters=256, kernel_size=7, padding='same')(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn_cat = []
    for i in range(num_tasks):
        nn2 = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(nn)
        nn2 = keras.layers.BatchNormalization()(nn2)
        nn2 = keras.layers.Activation('relu')(nn2)
        nn2 = keras.layers.Dropout(0.1)(nn2)
        nn2 = keras.layers.Dense(1, activation='softplus')(nn2)
        nn_cat.append(nn2)
    outputs = tf.concat(nn_cat, axis=2)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

############################################################
# layers and helper functions
############################################################
def conv_block(inputs, filters=None, kernel_size=1, activation='relu', activation_end=None,
               strides=1, dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
               pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None, bn_type='standard',
               kernel_initializer='he_normal', padding='same', w1=False):
    """Construct a single convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters:       Conv1D filters
      kernel_size:   Conv1D kernel_size
      activation:    relu/gelu/etc
      strides:       Conv1D strides
      dilation_rate: Conv1D dilation rate
      l2_scale:      L2 regularization weight.
      dropout:       Dropout rate probability
      conv_type:     Conv1D layer type
      residual:      Residual connection boolean
      pool_size:     Max pool width
      batch_norm:    Apply batch normalization
      bn_momentum:   BatchNorm momentum
      bn_gamma:      BatchNorm gamma (defaults according to residual)

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == 'separable':
        conv_layer = tf.keras.layers.SeparableConv1D
    elif w1:
        conv_layer = tf.keras.layers.Conv2D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    current = activate(current, activation)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = 'zeros' if residual else 'ones'
        if bn_type == 'sync':
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(
            momentum=bn_momentum,
            gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    # end activation
    if activation_end is not None:
        current = activate(current, activation_end)

    # Pool
    if pool_size > 1:
        if w1:
            current = tf.keras.layers.MaxPool2D(
                pool_size=pool_size,
                padding=padding)(current)
        else:
            current = tf.keras.layers.MaxPool1D(
                pool_size=pool_size,
                padding=padding)(current)

    return current


def conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
    """Construct a reducing convolution block.

    Args:
      inputs:        [batch_size, seq_length, features] input sequence
      filters_init:  Initial Conv1D filters
      filters_mult:  Multiplier for Conv1D filters
      repeat:        Conv block repetitions

    Returns:
      [batch_size, seq_length, features] output sequence
    """

    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    for ri in range(repeat):
        # convolution
        current = conv_block(current,
                             filters=int(np.round(rep_filters)),
                             **kwargs)

        # update filters
        rep_filters *= filters_mult

    return current, rep_filters


def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2,
                     conv_type='standard', dropout=0, repeat=1, round=False, **kwargs):
    """Construct a residual dilated convolution block.
    """

    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # dilate
        current = conv_block(current,
                             filters=filters,
                             kernel_size=kernel_size,
                             dilation_rate=int(np.round(dilation_rate)),
                             conv_type=conv_type,
                             bn_gamma='ones',
                             **kwargs)

        # return
        current = conv_block(current,
                             filters=rep_input.shape[-1],
                             dropout=dropout,
                             bn_gamma='zeros',
                             **kwargs)

        # residual add
        current = tf.keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult
        if round:
            dilation_rate = np.round(dilation_rate)

    return current

# depracated, poorly named
def dense_layer(inputs, units, activation='linear', kernel_initializer='he_normal',
                l2_scale=0, l1_scale=0, **kwargs):
    # apply dense layer
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(inputs)

    return current

def activate(current, activation, verbose=False):
    if verbose: print('activate:', activation)
    if activation == 'relu':
        current = tf.keras.layers.ReLU()(current)
    elif activation == 'polyrelu':
        current = PolyReLU()(current)
    elif activation == 'gelu':
        current = GELU()(current)
    elif activation == 'sigmoid':
        current = tf.keras.layers.Activation('sigmoid')(current)
    elif activation == 'tanh':
        current = tf.keras.layers.Activation('tanh')(current)
    elif activation == 'exponential':
        current = tf.keras.layers.Activation('exponential')(current)
    elif activation == 'softplus':
        current = Softplus()(current)
    else:
        print('Unrecognized activation "%s"' % activation)
        exit(1)

    return current

class GELU(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

def conv_layer(inputs, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, l2=None):
    """Convolutional layer with batchnorm and dropout"""
    if not l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    conv1 = keras.layers.Conv1D(filters=num_filters,
                                kernel_size=kernel_size,
                                strides=1,
                                activation=None,
                                use_bias=False,
                                padding=padding,
                                kernel_initializer='glorot_normal',
                                kernel_regularizer=l2,
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                bias_constraint=None,
                                )(inputs)
    conv1_bn = keras.layers.BatchNormalization()(conv1)
    conv1_active = keras.layers.Activation(activation)(conv1_bn)
    conv1_dropout = keras.layers.Dropout(dropout)(conv1_active)
    return conv1_dropout

def dilated_residual_block(input_layer, num_filters, filter_size, activation='relu', l2=None):
    """ dilated residual block composed of 3 sub-blocks of 1D conv, batchnorm, activation, dropout"""
    if not l2:
        l2 = keras.regularizers.l2(l2)
    else:
        l2 = None

    residual_conv1 = keras.layers.Conv1D(filters=num_filters,
                                         kernel_size=filter_size,
                                         strides=1,
                                         activation=None,
                                         use_bias=False,
                                         padding='same',
                                         dilation_rate=2,
                                         kernel_initializer='glorot_normal',
                                         kernel_regularizer=l2
                                         )(input_layer)
    residual_conv1_bn = keras.layers.BatchNormalization()(residual_conv1)
    residual_conv1_active = keras.layers.Activation('relu')(residual_conv1_bn)
    residual_conv1_dropout = keras.layers.Dropout(0.1)(residual_conv1_active)
    residual_conv2 = keras.layers.Conv1D(filters=num_filters,
                                         kernel_size=filter_size,
                                         strides=1,
                                         activation=None,
                                         use_bias=False,
                                         padding='same',
                                         dilation_rate=4,
                                         kernel_initializer='glorot_normal',
                                         kernel_regularizer=l2
                                         )(residual_conv1_dropout)
    residual_conv2_bn = keras.layers.BatchNormalization()(residual_conv2)
    residual_conv2_active = keras.layers.Activation('relu')(residual_conv2_bn)
    residual_conv2_dropout = keras.layers.Dropout(0.1)(residual_conv2_active)
    residual_conv3 = keras.layers.Conv1D(filters=num_filters,
                                         kernel_size=filter_size,
                                         strides=1,
                                         activation=None,
                                         use_bias=False,
                                         padding='same',
                                         dilation_rate=8,
                                         kernel_initializer='glorot_normal',
                                         kernel_regularizer=l2
                                         )(residual_conv2_dropout)
    residual_conv3_bn = keras.layers.BatchNormalization()(residual_conv3)
    residual_sum = keras.layers.add([input_layer, residual_conv3_bn])
    return keras.layers.Activation(activation)(residual_sum)

def residual_block(input_layer, kernel_size=3, activation='relu', num_layers=5, dropout=0.1):
    factor = range(1,num_layers)
    base_rate = 2
    """dilated residual convolutional block"""
    filters = input_layer.shape.as_list()[-1]
    nn = keras.layers.Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             activation=None,
                             padding='same',
                             dilation_rate=1)(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    for i in factor:
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv1D(filters=filters,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 dilation_rate=base_rate ** i)(nn)
        nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.add([input_layer, nn])
    return keras.layers.Activation(activation)(nn)

def early_stopping(patience = 10, verbose = 1):
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                             patience=patience,
                                             verbose=verbose,
                                             mode='min',
                                             baseline=None, restore_best_weights=True)
    return earlystop

def model_checkpoint(save_path,save_best_only = True):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=save_best_only,
                                                mode='min')
    return checkpoint

def reduce_lr(patience = 3,factor = 0.2):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=factor,
                                                 patience=patience,
                                                 min_lr=1e-7,
                                                 mode='min',
                                                 verbose=1)
    return reduce_lr
