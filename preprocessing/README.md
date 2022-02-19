# groundhog data preparation

This directory contains scripts for converting bigwig files into tfrecord files that are optimized for speeding up training with tensorflow. 

quant_dataset_tutorial.ipynb notebook contains the steps for making a small dataset and a description of the main script arguments to adjust. The last part of the notebook contains the steps for going from tfrecord files to bigwig tracks which is useful especially for visualizing the predictions of a model. 
