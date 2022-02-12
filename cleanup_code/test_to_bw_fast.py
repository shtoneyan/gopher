#!/usr/bin/env python
import util
import os, shutil
import numpy as np
import csv
import pyBigWig
import tensorflow as tf
from modelzoo import GELU
import metrics
import loss
import custom_fit
import time, sys
from scipy import stats
from loss import *
import yaml, glob
import subprocess
import gzip
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
import wandb


def enforce_constant_size(bed_path, output_path, window, compression=None):
    """generate a bed file where all peaks have same size centered on original peak"""

    # load bed file

    df = pd.read_csv(bed_path, sep=' ', header=None, compression=compression)

    df.columns = [0, 1, 2]
    chrom = df[0].to_numpy().astype(str)
    start = df[1].to_numpy()
    end = df[2].to_numpy()
    #print('# bed coordinates', len(end))
    # calculate center point and create dataframe
    middle = np.round((start + end)/2).astype(int)
    half_window = np.round(window/2).astype(int)

    # calculate new start and end points
    start = middle - half_window
    end = middle + half_window

    # filter any negative start positions
    data = {}
    for i in range(len(df.columns)):
        data[i] = df[i].to_numpy()
    data[1] = start
    data[2] = end
    #print('# coordinates after removing negatives', len(end))
    # create new dataframe
    df_new = pd.DataFrame(data);
#     print(df_new[df_new[1]<0])
    df_new = df_new[df_new.iloc[:,1] > 0]
    df_new = df_new[df_new.iloc[:,2] > 0]
    # save dataframe with fixed width window size to a bed file
    df_new.to_csv(output_path, sep='\t', header=None, index=False)

def change_filename(filepath, new_binningsize=None, new_thresholdmethod=None):
    '''This funciton switches between filenames used for bw files'''
    filename = os.path.basename(filepath) # extract file name from path
    directory = filepath.split(filename)[0] # extract folder name
    # split filename into variables
    celline, bigwigtype, bin, threshold = filename.split('.bw')[0].split('_')
    if new_binningsize != None: # if new bin size provided replace
        bin = new_binningsize
    if new_thresholdmethod != None: # if new threshold provided replace
        threshold = new_thresholdmethod
    # construct new filename
    new_filename = '_'.join([celline, bigwigtype, bin, threshold])+'.bw'
    return os.path.join(directory, new_filename) # return full path

def read_dataset(data_path, return_stats=False, batch_size=64):
    '''This function returns testset and corresponding cell lines'''
    # data_path = 'datasets/only_test/complete/random_chop/i_2048_w_1' - test set
    sts = util.load_stats(data_path) # load stats file
    # make dataset from tfrecords
    testset = util.make_dataset(data_path, 'test', sts, coords=True,
                                batch_size=batch_size, shuffle=False)
    targets_path = os.path.join(data_path, 'targets.txt') # load cell line names
    targets = pd.read_csv(targets_path, delimiter='\t')['identifier'].values
    if return_stats:
        return testset, targets, sts
    else:
        return testset, targets # return test set and cell line names

def get_config(run_path):
    '''This function returns config of a wandb run as a dictionary'''
    config_file = os.path.join(run_path, 'files', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_model(run_path, compile_model=True):
    '''This function loads a per-trained model'''
    config = get_config(run_path) # load wandb config
    if 'bin_size' in config.keys():
        bin_size = config['bin_size']['value'] # get bin size
    else:
        bin_size = 'NA'
    model_path = os.path.join(run_path, 'files', 'best_model.h5') # pretrained model
    # load model
    trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    if compile_model:
        loss_fn_str = config['loss_fn']['value'] # get loss
        loss_fn = eval(loss_fn_str)() # turn loss into function
        trained_model.compile(optimizer="Adam", loss=loss_fn)
    return trained_model, bin_size # model and bin size

def read_chrom_size(chrom_size_path):
    '''Load chromosome size file'''
    chrom_size = {}
    with open(chrom_size_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for line in rd:
            chrom_size[line[0]]=int(line[1])
    return chrom_size

def open_bw(bw_filename, chrom_size_path):
    '''This function opens a new bw file'''
    assert not os.path.isfile(bw_filename), 'Bw at {} alread exists!'.format(bw_filename)
    chrom_sizes = read_chrom_size(chrom_size_path) # load chromosome sizes
    bw = pyBigWig.open(bw_filename, "w") # open bw
    bw.addHeader([(k, v) for k, v in chrom_sizes.items()], maxZooms=0)
    return bw # bw file

def get_vals_per_range(bw_path, bed_path):
    '''This function reads bw (specific ranges of bed file) into numpy array'''
    bw = pyBigWig.open(bw_path)
    bw_list = []
    for line in open(bed_path):
        cols = line.strip().split()
        vals = bw.values(cols[0], int(cols[1]), int(cols[2]))
        bw_list.append(vals)
    bw.close()
    return bw_list

def remove_nans(all_vals_dict):
    '''This function masks nans in all values in a dict'''
    for i,(k, v) in enumerate(all_vals_dict.items()): # for each k, v
        if np.isnan(v).sum()>0: # if any nans present
            if 'nan_mask' not in locals(): # if nan_mask variable not created
                nan_mask = ~(np.isnan(v)) # make a variable nan_mask
            else:  # if already present
                nan_mask *= ~(np.isnan(v)) # add to the existing nan mask
    nonan_dict = {} # clean dictionary
    for k,v in all_vals_dict.items(): # for each k, v in original dict
        if 'nan_mask' in locals(): # if nans were found in any v
            nonan_dict[k] = v[nan_mask] # return the original masking out nans
        else: # if no nans were present
            nonan_dict[k] = v # just return original values
    return nonan_dict # return filtered dict of values

def remove_np_nans(all_vals_dict):
    assert 'pred' in all_vals_dict.keys() and 'truth' in all_vals_dict.keys(), 'Wrong keys!'
    include_rows = np.ones((all_vals_dict['pred'].shape[0]), dtype=bool)
    both_arrays = np.concatenate((all_vals_dict['pred'], all_vals_dict['truth']), axis=1)
    include_rows *= ~np.isnan(both_arrays).any(axis=1)
    all_vals_dict['pred'] = all_vals_dict['pred'][include_rows]
    all_vals_dict['truth'] = all_vals_dict['truth'][include_rows]
    return all_vals_dict

def make_truth_pred_bws(truth_bw_filename_suffix, pred_bw_filename_suffix,
                        bed_filename_suffix, testset, trained_model, bin_size,
                        cell_line_names, chrom_size_path, run_dir):
    '''This function makes ground truth and prediction bw-s from tfrecords dataset'''
    # open bw and bed files
    bedfiles = {}
    pred_bws = {}
    truth_bws = {}
    cell_line_N = len(cell_line_names)
    for cell_line, cell_line_name in enumerate(cell_line_names):
    # for cell_line, cell_line_name in enumerate(['A549']):
        # cell_line = 8
        output_dir = util.make_dir(os.path.join(run_dir, str(cell_line) + '_' + cell_line_name))
        pred_bw_filename = os.path.join(output_dir, cell_line_name + pred_bw_filename_suffix)
        pred_bws[cell_line] = open_bw(pred_bw_filename, chrom_size_path)
        truth_bw_filename = os.path.join(output_dir, cell_line_name + truth_bw_filename_suffix)
        truth_bws[cell_line] = open_bw(truth_bw_filename, chrom_size_path)
        bed_filename = os.path.join(output_dir, cell_line_name + bed_filename_suffix)
        bedfiles[cell_line] = open(bed_filename, "w")
    # go through test set data points
    for C, X, Y in testset: #per batch
        C = [str(c).strip('b\'').strip('\'') for c in C.numpy()] # coordinates
        P = trained_model(X) # make batch predictions
        for i, pred in enumerate(P): # per batch element
            chrom, start, end = C[i].split('_') # get chr, start, end
            start = int(start) # to feed into bw making function
            # for cell_line in [8]: # per cell line
            for cell_line in range(cell_line_N): # per cell line
                # write to ground truth file
                truth_bws[cell_line].addEntries(chrom, start,
                    values=np.array(np.squeeze(Y[i,:,cell_line]), dtype='float64'),
                    span=1, step=1)
                # write to prediction bw file
                pred_bws[cell_line].addEntries(chrom, start,
                    values=np.array(np.squeeze(pred[:,cell_line]), dtype='float64'),
                    span=bin_size, step=bin_size)
                # write ti bedfile (same for each cell line but needed for later)
                bedfiles[cell_line].write('{}\t{}\t{}\n'.format(chrom, start, end))
    # close everything
    # for cell_line in [8]:
    for cell_line in range(cell_line_N):

        truth_bws[cell_line].close()
        pred_bws[cell_line].close()
        bedfiles[cell_line].close()

def merge_bed(in_bed_filename):
    '''This function merges bed consequtive bed ranges'''
    split_filename = in_bed_filename.split('/') # deconstruct file name
    # rejoin file name with prefic 'merge'
    in_bed_filename_merged = '/'.join(split_filename[:-1] + ['merged_' + split_filename[-1]])
    # str command line for bedtools merge
    bashCmd = 'bedtools merge -i {} > {}'.format(in_bed_filename, in_bed_filename_merged)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    return in_bed_filename_merged # return new filename

def get_conc_pr(list1, list2):
    '''This function flattens np arrays and computes pearson r'''
    pr = stats.pearsonr(np.concatenate(list1), np.concatenate(list2))[0]
    assert ~np.isnan(pr)
    return pr

def get_per_seq_pr(bw_1, bw_2):
    res = []
    assert bw_1.shape == bw_2.shape, 'Unequal length bw lists!'
    for l in range(len(bw_1)):
        pr = stats.pearsonr(bw_1[l], bw_2[l])[0]
        res.append(pr)
    return res

def scipy_get_pr(bw_paths, bedfile='/home/shush/profile/QuantPred/bin_exp/truth/merged_truth_1_raw.bed'):
    '''This function computes pearson r from two bigwig files'''
    all_vals_dict_nans = {} # dictionary of all values
    for bw_path in bw_paths: # for each bw
        vals = get_vals_per_range(bw_path, bedfile) # convert bw to list of vals
        # convert to flattened np array
        all_vals_dict_nans[bw_path] = np.array([v  for v_sub in vals for v in v_sub])
        # remove nans
        all_vals_dict_1d = remove_nans(all_vals_dict_nans)
    # make sure there's enough values left
    assert len(all_vals_dict_1d[bw_paths[0]])>1 and len(all_vals_dict_1d[bw_paths[1]])>1, bw_paths
    # get pearson r
    pr = stats.pearsonr(all_vals_dict_1d[bw_paths[0]], all_vals_dict_1d[bw_paths[1]])[0]
    assert ~np.isnan(pr), 'Pearson R is nan for these {}'.format(bw_paths)
    return pr

def np_poiss(y_true, y_pred):
    def filter_fin(ar):
        return ar[np.isfinite(ar)]
    pois = y_pred - y_true * np.log(y_pred)
    return np.mean(filter_fin(pois))

def get_metrics(y_true, y_pred, bw_type, seq_len=2048):
    assert bw_type == 'raw' or bw_type == 'idr', 'Incorrect bw type!'
    if bw_type == 'raw':
        pr = get_conc_pr(y_true, y_pred)
    else:
        pr_list = get_per_seq_pr(y_true, y_pred)
        pr = np.mean(pr_list)
    assert y_true.shape[1] == seq_len, 'MSE calculation issue!'
    mse = mean_squared_error(y_true.T, y_pred.T)
    poiss_nll = np_poiss(y_true, y_pred)
    js_dist = distance.jensenshannon(y_true.T, y_pred.T)
    print(js_dist.shape)
    mean_js = np.nanmean(js_dist)
    return [pr, mse, poiss_nll, mean_js]


def get_replicates(cell_line_name, repl_labels = ['r2', 'r12'],
                    basenji_samplefiles=['/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basenji_sample_r2_file.tsv', '/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basenji_sample_r1,2_file.tsv']):
    '''This function finds paths to replicates for a specific cell line'''
    replicate_filepaths = {} #filename dictionary
    # per samplefile of replicates
    for b, basenji_samplefile in enumerate(basenji_samplefiles):
        # read in the samplefile
        basenji_samplefile_df = pd.read_csv(basenji_samplefile, sep='\t')
        # get the row with cell line name
        cell_row = basenji_samplefile_df[basenji_samplefile_df['identifier']==cell_line_name]['file']
        # make sure no duplicates detected
        assert not(len(cell_row) > 1), 'Multiple cell lines detected!'
        if len(cell_row) == 1: # if cell line replicate found
            replicate_filepaths[repl_labels[b]] = cell_row.values[0] # get filepath
    return replicate_filepaths # return dict of repl type and path

def get_idr(cell_line_name, idr_filename,
            basset_samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv',
           range_size = 2048,
           unmap_bed='/home/shush/genomes/GRCh38_unmap.bed'):
    '''This function makes cell line specific IDR file with constant window size for test set'''

    # make bed filename to be used in pearson r calculation with no merging of ranges
    split_filename = idr_filename.split('/')
    window_enf_idr = '/'.join(split_filename[:-1]+[split_filename[-1].split('.bed')[0]+ '_const' +'.bed'])
    nan_window_enf_idr = window_enf_idr + 'nan'
    # read in IDR samplefile
    basset_samplefile_df=pd.read_csv(basset_samplefile, sep='\t', header=None)
    # find cell line specific IDR file
    idr_file_gz = basset_samplefile_df[basset_samplefile_df[0]==cell_line_name][1].values[0]
    # str of command line command to filter test set peaks for cell line into new bed file
    interm_bed = '{}_idr_strict_peaks.bed'.format(cell_line_name)
    make_bedfile = "scp {} temp.bed.gz; gunzip temp.bed.gz; grep chr8 temp.bed|awk '{{print $1, $2, $3}}'|sort -k1,1 -k2,2n|uniq > {}; rm temp.bed".format(idr_file_gz, interm_bed)
    process = subprocess.Popen(make_bedfile, shell=True)
    output, error = process.communicate()
    # make new bedfile with constant bed ranges
    enforce_constant_size(interm_bed, nan_window_enf_idr, range_size)
    # remove regions partially in unmap regions
    filter_bed = 'bedtools intersect -v -a {} -b {} > {}'.format(nan_window_enf_idr, unmap_bed, window_enf_idr)
    process = subprocess.Popen(filter_bed, shell=True)
    output, error = process.communicate()
    # merge ranges so that bw writing can happen later
    merge_bed = 'bedtools merge -i {} > {}; rm {}; rm {}'.format(window_enf_idr, idr_filename, interm_bed, nan_window_enf_idr)
    process = subprocess.Popen(merge_bed, shell=True)
    output, error = process.communicate()

def bw_from_ranges(in_bw_filename, in_bed_filename, out_bw_filename,
                   chrom_size_path, bin_size=1, threshold=-1,
                   out_bed_filename=''):
    '''
    This function creates bw file from existing bw file but only from specific
    bed ranges provided in the bed file, and optionally thresholds the bed file
    as well as optionally outputs the regions selected if out_bed_filename provided
    '''
    if len(out_bed_filename) > 0: # if out_bed_filename given to save recorded ranges
        bedfile = open(out_bed_filename, "w") # open new bed file
    in_bw = pyBigWig.open(in_bw_filename) # open existing bw
    out_bw = open_bw(out_bw_filename, chrom_size_path) # open new bw
    in_bedfile = open(in_bed_filename) # open existing bed file
    for line in in_bedfile: # per bed range
        cols = line.strip().split()
        vals = in_bw.values(cols[0], int(cols[1]), int(cols[2])) # get coords
        vals = np.array(vals, dtype='float64') # get values
        if np.max(vals) > threshold: # if above threshold
            # write values to new bw using bin size as step
            vals = vals.reshape(len(vals)//bin_size, bin_size).mean(axis=1)
            out_bw.addEntries(cols[0], int(cols[1]), values=vals, span=bin_size,
                              step=bin_size)
            if len(out_bed_filename) > 0: # if bed file of ranges needed
                bedfile.write(line) # record range above threshold
    # close files
    in_bw.close()
    out_bw.close()
    if len(out_bed_filename) > 0:
        bedfile.close()

def avg_cov(np_array):
    return np.mean(np_array, axis=1)

def save_jointplot(cov_dict, save_path, cell_line_name):
    if 'raw' in save_path:
        title_add = 'raw'
        c = 'blue'
        a = 0.1
    else:
        title_add = 'IDR'
        c = 'purple'
        a = 0.2
    mean_cov = pd.DataFrame({k:avg_cov(cov_dict[k]) for k in ['truth', 'pred']})
    mean_cov.columns = ['ground truth', 'prediction']
    joint_grid = sns.jointplot(data=mean_cov, x='ground truth', y='prediction', color=c,
                               kind="reg", joint_kws = {'scatter_kws':dict(alpha=a)})
    x0, x1 = joint_grid.ax_joint.get_xlim()
    y0, y1 = joint_grid.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    joint_grid.ax_joint.plot(lims, lims, '--k')
    joint_grid.fig.subplots_adjust(top=0.9)
    _=joint_grid.fig.suptitle(cell_line_name+' mean coverage, ' + title_add)
    plt.savefig(save_path)

def scatter_cell_line_prs(corr_dfs, savedir):
    melted_corrs = {}
    for bw_set in ['truth', 'pred']:
        df = corr_dfs[bw_set]
        np.tril(np.ones(df.shape)).astype(np.bool)
        df_lt = df.where(~np.tril(np.ones(df.shape)).astype(np.bool))
        df_lt = df_lt.stack().reset_index()
        df_lt.columns = ['cell_line1', 'cell_line2' ,'pearson\'s r']
        melted_corrs[bw_set] = df_lt
    avg_df = melted_corrs['truth'].merge(melted_corrs['pred'], on=['cell_line1', 'cell_line2'], suffixes=[' true', ' pred'])
#     avg_df.to_csv(os.path.join(outdir, 'correlation_scatterplot_{}_{}.csv'.format(set_type, model_id)))
    ax = sns.scatterplot(data=avg_df, x='pearson\'s r true', y='pearson\'s r pred')
    x = [i/10 for i in range(2, 11)]
    plt.plot(x, x, 'r--')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.savefig(os.path.join(savedir, 'correlation_scatterplot.svg'))

def get_corr_df(avg_cov_all, cell_line_names):
    corr_dfs = {}
    for bw_source in ['truth', 'pred']:
        corr_df = pd.DataFrame(avg_cov_all[bw_source]).T.corr()
        corr_df.columns = cell_line_names
        corr_df['cell_lines'] = cell_line_names
        corr_df.set_index('cell_lines', inplace=True)
        corr_dfs[bw_source] = corr_df
    return corr_dfs

def plot_corr_matrices(corr_dfs, savedir):
    fig, axs = plt.subplots(1, 2, figsize=[20, 8])
    min_lim = pd.concat([v for _, v in corr_dfs.items()]).min().min()
    titles = ['ground truth', 'prediction']
    truth_heatmap = sns.heatmap(corr_dfs['truth'], annot=True, vmin=min_lim, vmax=1,  ax=axs[0], cmap='flare')
    axs[0].set_title(titles[0])
    pred_heatmap = sns.heatmap(corr_dfs['pred'], annot=True, vmin=min_lim, vmax=1, ax=axs[1], cmap='flare')
    axs[1].set_title(titles[1])
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'correlation_matrices.svg'))


def process_run(run_path,
                      threshold=2,
                      data_path='datasets/chr8/complete/random_chop/i_2048_w_1',
                      chrom_size_path="/home/shush/genomes/GRCh38_EBV.chrom.sizes.tsv",
                      get_replicates=False,
                      bigwig_foldername='bigwigs'):
    '''This function processes a wandb run and outputs bws of following types:
    - ground truth base resolution
    - ground truth binned if model is trained on binned dataset
    - prediction binned at whatever model is trained on
    - ground truth and prediction thresholded using provided threshold (optional)
    - ground truth and pred of IDR peaks per cell line (cut from full bw not
      predicted anew) (optional)
    - ground truth of replicates in every filtering type (optional)'''
    testset, targets = read_dataset(data_path) # get dataset
    trained_model, bin_size = read_model(run_path) # get model
    # set filename for base res ground truth
    truth_bw_filename_suffix = "_truth_1_raw.bw"
    # set filename for prediction bw at model resolution
    pred_bw_filename_suffix = "_pred_{}_raw.bw".format(bin_size)
    bed_filename_suffix = "_truth_1_raw.bed" # filename for bed file of ranges
    # new subfolder to save bigwigs in
    run_subdir = util.make_dir(os.path.join(run_path, bigwig_foldername))
    # make ground truth, pred bigwigs and bed file of ranges where dataset is
    # for each cell line in a separate subdir in run_subdir
    print('Making ground truth and prediction bigwigs')
    t0 = time.time()
    make_truth_pred_bws(truth_bw_filename_suffix, pred_bw_filename_suffix, bed_filename_suffix,
                          testset, trained_model, bin_size, targets,
                          chrom_size_path, run_subdir)

    t1 = time.time()
    print('Time = {}mins'.format((t1-t0)//60))
    for subdir in tqdm(os.listdir(run_subdir)): # per cell line directory
        print(subdir)
        output_dir = os.path.join(run_subdir, subdir) # cell line full path
        subdir_split = subdir.split('_') # split into id and cell line name
        # make sure no other file is detected
        assert len(subdir_split) == 2, 'Check subdirectory names for underscores!'
        cell_line_index, cell_line_name = subdir_split # read in id and name
        # define base res ground truth path
        bed_filename = os.path.join(output_dir, cell_line_name + '_truth_1_raw.bed')
        # define IDR bw path
        cell_line_truth_idr = os.path.join(output_dir, cell_line_name + '_truth_1_idr.bed')
        get_idr(cell_line_name, cell_line_truth_idr) # find cell line IDR bed file
        print('Processing cell line '+cell_line_name)
        #### make nonthresholded non binned replicates
        rX_bw_filenames = []
        if get_replicates: # if replicates needed
            replicate_filepaths = get_replicates(cell_line_name) # find em
            for rX, rX_bw_path in replicate_filepaths.items(): # per replicate
                # make a new base res bigwig path
                out_rX = os.path.join(output_dir, cell_line_name + '_{}_1_raw.bw'.format(rX))
                # save bw filename for later binning and thresholding
                rX_bw_filenames.append(out_rX)
                # extract bw values same as the base res ground truth bw
                bw_from_ranges(rX_bw_path, bed_filename, out_rX, chrom_size_path)
        #### bin ground truth, replicates (rXs = r2, r12, etc.)
        truth_bw_filename = os.path.join(output_dir, cell_line_name+truth_bw_filename_suffix)
        # new binned filenames which may be original ones if bin size = 1 or
        # new one if bin size more than 1
        binned_filenames = []
        # for each newly made bw except for prediction (which is already binned
        # to whatever we need)
        for in_bw in rX_bw_filenames+[truth_bw_filename]:
            # make new filename
            out_bw = change_filename(in_bw, new_binningsize=str(bin_size))
            binned_filenames.append(out_bw) # save for later
            if bin_size != 1: # if bin less than one don't redo bw making!
                bw_from_ranges(in_bw, bed_filename, out_bw, chrom_size_path, bin_size=bin_size)
        # add pred to the binned bw filename set
        pred_bw_filename = os.path.join(output_dir, cell_line_name+pred_bw_filename_suffix)
        print(binned_filenames+[pred_bw_filename])
        #### threshold all using IDR file of ground truth bw
        for binned_filename in binned_filenames+[pred_bw_filename]: # for all new bws
            # make IDR bw
            out_bw = change_filename(binned_filename, new_thresholdmethod='idr')
            # filter IDR peak regions only
            bw_from_ranges(binned_filename, cell_line_truth_idr, out_bw, chrom_size_path)

        # if threhsold given, threshold all using absolute threshold
        if threshold > 0:
            # new bed filename
            thresh_str = 'thresh'+str(threshold)
            thresh_bedfile = truth_bw_filename.split('.bw')[0]+'_{}.bed'.format(thresh_str)
            # new bw filename for ground truth
            truth_thresh_filename = change_filename(truth_bw_filename, new_thresholdmethod=thresh_str)
            bw_from_ranges(truth_bw_filename, bed_filename, truth_thresh_filename, chrom_size_path, threshold=threshold, out_bed_filename=thresh_bedfile)
            # for all binned bws that are to be thresholded
            for binned_filename in binned_filenames+[pred_bw_filename]:
                print(binned_filename)
                out_thresh = change_filename(binned_filename, new_thresholdmethod=thresh_str)
                if 'truth_1_thresh' not in out_thresh: # this one would already be made above
                    bw_from_ranges(binned_filename, thresh_bedfile, out_thresh, chrom_size_path)

def evaluate_run_performance(run_dir, rm_bws=False, rm_all=False):
    bin_size = get_config(run_dir)['bin_size']['value']
    loss_fn = get_config(run_dir)['loss_fn']['value']
    model_fn = get_config(run_dir)['model_fn']['value']
    # get the cell line specific directory
    bigwigs_dir = os.path.join(run_dir, 'bigwigs')
    folders_of_cell_lines = [os.path.join(bigwigs_dir, f) for f in os.listdir(bigwigs_dir) if os.path.isdir(os.path.join(bigwigs_dir, f))]
    summary_dir = util.make_dir(os.path.join(run_dir, 'summary'))
    summary_lines = []
    summary_line_cols = ['run_dir', 'model_fn', 'loss_fn', 'bin_size', 'cell_line_name',
                          'raw pearson r', 'raw MSE', 'raw poisson NLL', 'raw JS'
                          'idr pearson r', 'idr MSE', 'idr poisson NLL', 'idr JS']
    avg_cov_all = {'raw':{'truth':[], 'pred':[]}}
    cell_line_names = [cell_line_dir.split('/')[-1].split('_')[1] for cell_line_dir in folders_of_cell_lines]
    df_path = os.path.join(summary_dir, 'summary_metrics.csv')
    for c, cell_line_dir in tqdm(enumerate(folders_of_cell_lines)):

        cell_line_name = cell_line_names[c]
        plot_paths = {bw_t: os.path.join(summary_dir, '{}_{}_scatter.svg'.format(bw_t, cell_line_name)) for bw_t in ['raw', 'idr']}
        print(cell_line_dir)
        summary_line = [run_dir, model_fn, loss_fn, bin_size, cell_line_name]
        t_bw, p_bw = [os.path.join(cell_line_dir, '{}_{}_{}_raw.bw'.format(cell_line_name, x, bin_size)) for x in ['truth', 'pred']]
        raw_bed = os.path.join(cell_line_dir, '{}_truth_1_raw.bed'.format(cell_line_name))
        idr_bed = os.path.join(cell_line_dir, '{}_truth_1_idr_const.bed'.format(cell_line_name))
        assert all([os.path.isfile(f) for f in [t_bw, p_bw, raw_bed, idr_bed]]), 'One or more of the files not found in {}'.format(cell_line_dir)
        np_cov_dict = {}
        for bw_type, bed_filename in [('raw', raw_bed), ('idr', idr_bed)]:
            np_cov_dict[bw_type] = {}
            for fold_type, bw_filename in [('truth', t_bw), ('pred', p_bw)]:
                np_cov_dict[bw_type][fold_type] = get_vals_per_range(bw_filename, bed_filename)
        for bw_type in ['raw', 'idr']:
            np_cov_dict[bw_type] = remove_np_nans(np_cov_dict[bw_type]) # clean up raw
            # get pearson r values, MSE, poisson
            summary_line += get_metrics(np_cov_dict[bw_type]['truth'], np_cov_dict[bw_type]['pred'], bw_type)
            save_jointplot(np_cov_dict[bw_type], plot_paths[bw_type], cell_line_name) # plot raw
            avg_cov_dict = {} # save avg for correlation matrix
            avg_cov_dict[bw_type] = {bw_source: avg_cov(np_cov_dict[bw_type][bw_source]) for bw_source in ['truth', 'pred']}
            if bw_type == 'raw':
                avg_cov_all[bw_type]['truth'].append(avg_cov_dict[bw_type]['truth'])
                avg_cov_all[bw_type]['pred'].append(avg_cov_dict[bw_type]['pred'])
        summary_lines.append(summary_line) # save summary metrics
    summary_df = pd.DataFrame(summary_lines, columns = summary_line_cols)
    summary_df.to_csv(df_path, index=None)
    corr_dfs = get_corr_df(avg_cov_all['raw'], cell_line_names)
    plot_corr_matrices(corr_dfs, summary_dir)
    plt.clf()
    scatter_cell_line_prs(corr_dfs, summary_dir)
    plt.clf()

    if rm_all:
        print('Cleaning bigwig folder!')
        os.rmdir(bigwigs_dir)
    elif rm_bws:
        print('Removing only bw files!')
        all_bw_list = glob.glob('/home/shush/profile/QuantPred/0run/bigwigs/*/*bw')
        for file in all_bw_list:
            os.remove(file)




def evaluate(run_path):
    process_run(run_path, threshold=0)
    # evaluate_run_performance(run_path)

if __name__ == '__main__':
    run_dir = sys.argv[1]
    print('Processing '+run_dir)
    evaluate(run_dir)
