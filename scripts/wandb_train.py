#!/usr/bin/env python
import json
import os
import h5py
import sys
import utils
import numpy as np
import tensorflow as tf
import modelzoo
import losses
import time
import wandb
import custom_fit

def fit_robust(model_name_str, loss_type_str, window_size, bin_size, data_dir,
               output_dir, config={}):
    """

    :param model_name_str: func name defined in the modelzoo.py
    :param loss_type_str: loss func name defines in losses.py
    :param window_size: input length for the model
    :param bin_size: bin resolution
    :param data_dir: dataset path
    :param output_dir: output where to save the model
    :param config: set of parameters for defining a run
    :return: training metrics history
    """

  default_config = {'num_epochs':30, 'batch_size':64, 'shuffle':True,
  'metrics':['mse','pearsonr', 'poisson'], 'es_start_epoch':1,
  'l_rate':0.001, 'es_patience':6, 'es_metric':'loss',
  'es_criterion':'min', 'lr_decay':0.3, 'lr_patience':10,
  'lr_metric':'loss', 'lr_criterion':'min', 'verbose' : True,
  'log_wandb':True,'rev_comp' : True,'crop' : True,
  'record_test':False, 'alpha':False,  'loss_params':[],
  'sigma':20}

  for key in default_config.keys():
      if key in config.keys():
          default_config[key] = config[key]
  print(data_dir)
  if '2048' in data_dir:
      rev_comp = False
      crop_window = True



  if not os.path.isdir(output_dir):
      os.mkdir(output_dir)

  optimizer = tf.keras.optimizers.Adam(learning_rate=default_config['l_rate'])
  model = eval('modelzoo.'+model_name_str) # get model function from model zoo
  output_len = window_size // bin_size


  loss = eval('loss.'+loss_type_str)(loss_params=default_config['loss_params'])

  trainset = util.make_dataset(data_dir, 'train', util.load_stats(data_dir), batch_size=default_config['batch_size'])
  validset = util.make_dataset(data_dir, 'valid', util.load_stats(data_dir), batch_size=59)

  json_path = os.path.join(data_dir, 'statistics.json')
  with open(json_path) as json_file:
    params = json.load(json_file)
  print(params['num_targets'])
  if loss_type_str == 'poisson' and model_name_str == 'bpnet':
    model = model((window_size, 4),(output_len, params['num_targets']), softplus = True, wandb_config=config)
  else:
    model = model((window_size, 4),(output_len, params['num_targets']), wandb_config=config)

  if not model:
    raise BaseException('Fatal filter N combination!')


  print(model.summary())
  train_seq_len = params['train_seqs']
  if model_name_str == 'ori_bpnet':
  # create trainer class
    trainer =custom_fit.RobustTrainer(model, loss, optimizer, window_size, bin_size, params['num_targets'], default_config['metrics'],
                                    ori_bpnet_flag = True,rev_comp=default_config['rev_comp'],crop=default_config['crop'],
                                    sigma = default_config['sigma'])
  else:
    trainer =custom_fit.RobustTrainer(model, loss, optimizer, window_size, bin_size, params['num_targets'],default_config['metrics'],
                                    ori_bpnet_flag = False,rev_comp=default_config['rev_comp'],crop=default_config['crop'],
                                    sigma = default_config['sigma'])

  # set up learning rate decay
  trainer.set_lr_decay(decay_rate=default_config['lr_decay'], patience=default_config['lr_patience'],
                        metric=default_config['lr_metric'], criterion=default_config['lr_criterion'])
  trainer.set_early_stopping(patience=default_config['es_patience'], metric=default_config['es_metric'],
                            criterion=default_config['es_criterion'])

  # train model
  for epoch in range(default_config['num_epochs']):
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    #Robust train with crop and bin
    trainer.robust_train_epoch(trainset,num_step=train_seq_len//default_config['batch_size']+1,
                                batch_size = default_config['batch_size'])

    # validation performance
    trainer.robust_evaluate('val', validset,
                            batch_size=default_config['batch_size'], verbose=default_config['verbose'])


    # check learning rate decay
    trainer.check_lr_decay('loss')

    # check early stopping
    if epoch >= default_config['es_start_epoch']:

      if trainer.check_early_stopping('val'):
        print("Patience ran out... Early stopping.")
        break
    if default_config['log_wandb']:
        # Logging with W&B
        current_hist = trainer.get_current_metrics('train')
        wandb.log(trainer.get_current_metrics('val', current_hist))

  # compile history
  history = trainer.get_metrics('train')
  history = trainer.get_metrics('val', history)
  model.save(os.path.join(output_dir, "best_model.h5"))
  return history

def train_config(config=None):
    """

    :param config:
    :return:
    """
  with wandb.init(config=config) as run:

    config = wandb.config
#     print(config.data_dir)
#     print(config.l_rate)

    history = fit_robust(config.model_fn, config.loss_fn,
                       config.window_size, config.bin_size, config.data_dir,
                       output_dir = wandb.run.dir, config=config)


def main():
  exp_id = sys.argv[1]
  exp_n = sys.argv[2]
  if 'sweeps' in exp_id:
      exp_id = '/'.join(exp_id.split('/sweeps/'))
  else:
      raise BaseException('Sweep ID invalid!')
  sweep_id = exp_id
  wandb.login()
  wandb.agent(sweep_id, train_config, count=exp_n)


# __main__
################################################################################
if __name__ == '__main__':
    main()
