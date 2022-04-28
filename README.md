
<img src="./crop_gh.png" width="100" height='100'>

**GOPHER**: **G**en**O**mic **P**rofile-model compre**H**ensive **E**valuato**R**

This repository contains scripts for data preprocessing, training deep learning models for DNA sequence to epigenetic function prediction and evaluation of models.

The repo contains a set of tutorial jupyter notebooks that illustrate these steps on a toy dataset. The two notebooks below are required prerequisites for the rest of tutorials:
- preprocessing/preprocessing/quant_dataset_tutorial.ipynb
- tutorials/train_model.ipynb


To replicate the results of the manuscript run the scripts in the analysis directory. As a prerequisite download and unzip dataset.zip, trained_models.zip from zenodo https://doi.org/10.5281/zenodo.6464031 within the git repo. These contain test sets and pre-trained models. The analysis scripts can be ran in any order as long as paper_run_evaluate.py is ran first, in order to produce model evaluations which is required for further steps.
