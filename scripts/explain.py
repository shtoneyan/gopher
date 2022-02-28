import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import subprocess
import os, shutil, h5py, scipy
import utils
import custom_fit
import seaborn as sns
import tfomics
from operator import itemgetter












def enforce_const_range(site, window):
    half_window = np.round(window / 2).astype(int)
    start = site - half_window
    end = site + half_window
    return start, end


def combine_beds(samplefile, out_path):
    bed_paths = pd.read_csv(samplefile, sep='\t', header=None)[1].values
    combined_csv = pd.concat([(pd.read_csv(f, sep='\t', header=None).iloc[:, :3]).drop_duplicates() for f in bed_paths])
    combined_csv.to_csv(out_path, sep='\t', header=None, index=None)


def filter_dataset(dsq_path, out_path='dsq_all.bed'):
    dsq_all = pd.read_csv(dsq_path, sep='\t')
    dsq_all['ID'] = dsq_all.index
    dsq_filt = dsq_all[(dsq_all['chrom'] == 'chr8')]
    dsq_filt[['a1', 'a2']] = dsq_filt['genotypes'].str.split('/', expand=True)  # write into separate columns
    dsq_filt.to_csv(out_path, sep='\t', header=False, index=None)
    return dsq_filt


def bed_intersect(dataset_bed, comb_peak, out_path):
    bashCmd = "bedtools intersect -wa -a {} -b {} > {}".format(dataset_bed, comb_peak, out_path)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    print(error)


def extend_ranges(column_names, bedfile, out_path, window):
    dsq_df = pd.read_csv(bedfile, sep='\t', header=None, index_col=None)
    dsq_df.columns = column_names  # list(dsq_filt)
    dsq_filt = dsq_df[['chrom', 'snpChromStart', 'snpChromEnd', 'a1', 'a2',
                       'strand', 'rsid', 'pred.fit.pctSig', 'ID']]
    # get the 3K range start and end
    start, end = enforce_const_range(dsq_filt['snpChromEnd'] - 1, window)
    dsq_ext = dsq_filt.copy()
    #
    dsq_ext.iloc[:, 1] = start.values
    dsq_ext.iloc[:, 2] = end.values
    dsq_nonneg = dsq_ext[dsq_ext['snpChromStart'] > 0]
    dsq_nonneg = dsq_nonneg.reset_index(drop=True)
    dsq_nonneg['counts'] = dsq_nonneg.groupby(['chrom', 'snpChromStart'])['snpChromStart'].transform('count').values
    dsq_nonneg = dsq_nonneg.drop_duplicates().reset_index(drop=True)
    dsq_nonneg.to_csv(out_path, header=None, sep='\t', index=None)
    counts_per_cell = dsq_nonneg['counts'].values
    pct_sign = dsq_nonneg['pred.fit.pctSig'].values
    return dsq_nonneg, counts_per_cell, pct_sign


def bed_to_fa(bedfile='test_ds.csv', out_fa='test_ds.fa',
              genome_file='/home/shush/genomes/hg19.fa'):
    bashCmd = "bedtools getfasta -fi {} -bed {} -s -fo {}".format(genome_file, bedfile, out_fa)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    print(error)


def str_to_onehot(coords_list, seqs_list, dsq_nonneg, window):
    N = len(seqs_list)
    mid = window // 2
    onehot_ref = np.empty((N, window, 4))
    onehot_alt = np.empty((N, window, 4))
    coord_np = np.empty((N, 4))  # chrom, start, end coordinate array

    for i, (chr_s_e, seq) in enumerate(zip(coords_list, seqs_list)):
        alt = ''
        strand = chr_s_e.split('(')[-1].split(')')[0]
        pos_dict = {'+': mid, '-': mid - 1}
        pos = pos_dict[strand]
        coord_np[i, 3] = pos_dict[strand] - mid - 1

        if seq[pos] == dsq_nonneg['a1'][i]:
            alt = dsq_nonneg['a2'][i]

        elif seq[pos] == dsq_nonneg['a2'][i]:
            alt = dsq_nonneg['a1'][i]
        else:
            break

        chrom, s_e = chr_s_e.split('(')[0].split(':')
        s, e = s_e.split('-')
        coord_np[i, :3] = int(chrom.split('chr')[-1]), int(s), int(e)

        onehot = dna_one_hot(seq)
        onehot_ref[i, :, :] = onehot

        onehot_alt[i, :, :] = onehot
        onehot_alt[i, mid, :] = dna_one_hot(alt)[0]

    return (onehot_ref, onehot_alt, coord_np)


def onehot_to_h5(onehot_ref, onehot_alt, coord_np, pct_sign, out_dir='.', filename='onehot.h5'):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    onehot_ref_alt = h5py.File(os.path.join(out_dir, filename), 'w')
    onehot_ref_alt.create_dataset('ref', data=onehot_ref, dtype='float32')
    onehot_ref_alt.create_dataset('alt', data=onehot_alt, dtype='float32')
    onehot_ref_alt.create_dataset('fasta_coords', data=coord_np, dtype='i')
    # onehot_ref_alt.create_dataset('cell_lines', data=cell_lines, dtype='i')
    onehot_ref_alt.create_dataset('pct_sign', data=pct_sign, dtype='float32')
    onehot_ref_alt.close()


def table_to_h5(dsq_path,
                samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv',
                out_peaks='combined_atac.bed', out_filt='dsq_all.bed',
                out_open='dsq_open.bed', out_fin='filt_open_ext.bed',
                out_fa='ext.fa', genome_file='/home/shush/genomes/hg19.fa',
                window=3072, out_dir='.', out_h5='onehot.h5', save_files=True):
    print('Combining IDR beds')
    combine_beds(samplefile, out_peaks)
    print('Filtering in test set chromosomes in the dataset ')
    column_names = filter_dataset(dsq_path, out_filt)
    print('Filtering SNPs in the open chromatin regions')
    bed_intersect(out_filt, out_peaks, out_open)
    print('Extending regions around the SNP')
    dsq_nonneg, counts_per_cell, pct_sign = extend_ranges(column_names, out_open, out_fin, window)
    print('Converting bed to fa')
    bed_to_fa(out_fin, out_fa, genome_file)
    print('converting fa to one hot encoding')
    coords_list, seqs_list = fasta2list(out_fa)
    onehot_ref, onehot_alt, coord_np = str_to_onehot(coords_list, seqs_list,
                                                     dsq_nonneg, window)
    print('Saving onehots as h5')
    onehot_to_h5(onehot_ref, onehot_alt, coord_np, counts_per_cell, pct_sign, out_dir, out_h5)

    interm_files = [out_peaks, out_filt, out_open, out_fin, out_fa]
    if save_files:
        for f in interm_files:
            dst_f = os.path.join(out_dir, f)
            shutil.move(f, dst_f)
    else:
        for f in interm_files:
            os.remove(f)


def merge_one_cell_line(chr8_dsq_file, idr_bed, i='0', out_merged_bed='merged.bed'):
    assert isinstance(i, str), 'Input str as identifier!'

    bashCmd = "bedtools intersect -wa -a {} -b {} > {}".format(chr8_dsq_file, idr_bed, out_merged_bed)
    process = subprocess.Popen(bashCmd, shell=True)
    output, error = process.communicate()
    chr8_dsq = pd.read_csv(chr8_dsq_file, header=None, sep='\t')
    # keep_cols = ['chrom', 'snpChromStart', 'snpChromEnd']
    keep_cols = ['chrom', 'snpChromStart', 'snpChromEnd', 'rsid',
                 'pred.fit.pctSig', 'strand', 'motifname',
                 'position', 'genotypes', 'ID', 'a1', 'a2']
    chr8_dsq.columns = keep_cols
    merged = pd.read_csv('merged.bed', header=None, sep='\t')
    merged.columns = keep_cols
    merged['idr_N'] = i
    merged_dsq = chr8_dsq.merge(merged, how='outer')
    open_vcfs = merged_dsq[merged_dsq['idr_N'] == i]
    open_vcfs = open_vcfs.drop_duplicates().reset_index(drop=True)
    return open_vcfs


def get_h5_with_cells(dsq_path, window=3072, out_dir='.',
                      samplefile='/mnt/906427d6-fddf-41bf-9ec6-c3d0c37e766f/amber/ATAC/basset_sample_file.tsv',
                      save_files=False):
    dsq = pd.read_csv(dsq_path, sep='\t')  # dataframe from the paper
    dsq_chr8 = filter_dataset(dsq_path, 'dsq_chr8.bed')  # filter chr8 VCFs
    # list to save per cell line vcfs in the open chromatin regions
    per_cell_open_vcfs = []
    # open samplefile and get IDR filepaths
    bed_paths = pd.read_csv(samplefile, sep='\t', header=None)[1].values
    # per cell line
    for f, bed_file in enumerate(bed_paths):
        # get VCFs in the open chromatin region
        open_vcfs = merge_one_cell_line('dsq_chr8.bed', bed_file, i=str(f))
        per_cell_open_vcfs.append(open_vcfs)  # save it
    # put all into one df
    conc_vcfs = pd.concat(per_cell_open_vcfs)
    # merge redundant ones and collect cell line info
    merged_vcfs = conc_vcfs.groupby(['chrom', 'snpChromStart', 'snpChromEnd'])['idr_N'].apply(', '.join).reset_index()
    # reattach columsn with metadata
    complete_df = merged_vcfs.merge(conc_vcfs, how='left', on=['chrom', 'snpChromStart', 'snpChromEnd'])
    # remove redundant columns and rows
    complete_unq = complete_df.drop(columns='idr_N_y').drop_duplicates().reset_index(drop=True)
    # get the 3K range start and end
    start, end = enforce_const_range(complete_unq['snpChromEnd'] - 1, window)
    complete_unq.insert(1, '3K_start', start.values)  # add starts
    complete_unq.insert(2, '3K_end', end.values)  # add ends
    complete_unq = complete_unq[complete_unq['3K_start'] > 0]  # remove ones starting at negative coords
    complete_unq.rename(columns={'idr_N_x': 'cell_lines'}, inplace=True)  # rename column
    complete_unq.to_csv(os.path.join(out_dir, 'vcf_metadata.csv'), sep='\t',
                        index=None)  # save the complete metadata table
    # save version needed for fa conversion
    complete_unq[['chrom', '3K_start', '3K_end', 'rsid', 'pred.fit.pctSig',
                  'strand']].to_csv('out_fin.bed', sep='\t', header=None, index=None)
    bed_to_fa('out_fin.bed', 'out.fa', genome_file='/home/shush/genomes/hg19.fa')
    coords_list, seqs_list = fasta2list('out.fa')
    onehot_ref, onehot_alt, coord_np = str_to_onehot(coords_list, seqs_list,
                                                     complete_unq, window)

    onehot_to_h5(onehot_ref, onehot_alt, coord_np,
                 np.array(complete_unq['pred.fit.pctSig'].values), out_dir)
    shutil.copy(dsq_path, os.path.join(out_dir, dsq_path))
    interm_files = ['dsq_chr8.bed', 'out_fin.bed', 'out.fa']
    if save_files:
        for f in interm_files:
            dst_f = os.path.join(out_dir, f)
            shutil.move(f, dst_f)
    else:
        for f in interm_files:
            os.remove(f)






def function_batch(X, fun, batch_size=128, **kwargs):
    """ run a function in batches """

    dataset = tf.data.Dataset.from_tensor_slices(X)
    outputs = []
    for x in dataset.batch(batch_size):
        f = fun(x, **kwargs)
        outputs.append(f)
    return np.concatenate(outputs, axis=0)

    def plot_true_pred(idx, idx_name, cell_line=13):
        fig, axs = plt.subplots(1, 2, figsize=[15, 5])
        axs[0].plot(filtered_Y[idx, :, cell_line].T);
        axs[1].plot(np.repeat(preds[idx, :, cell_line], 32, axis=1).T);
        pr = []
        for i in idx:
            pr.append(pearsonr(filtered_Y[i, :, cell_line], np.repeat(preds[i, :, cell_line], 32))[0])
        pr = np.round(np.mean(pr), 3)

        plt.suptitle('{} points, mean per seq pearson r = {}'.format(idx_name, pr));


def plot_embedding(embedding, cluster_index):
    fig = plt.figure(figsize=[10, 10])
    sns.scatterplot(
        x=embedding[:, 1],
        y=embedding[:, 0],
        alpha=0.2
    )

    sns.scatterplot(
        x=embedding[cluster_index, 1],
        y=embedding[cluster_index, 0],
        alpha=1
    )


def plot_embedding_with_box(embedding, anchors, w=1.2, h=1.5, colors=['r', 'g', 'b']):
    fig, ax = plt.subplots(1, 1, figsize=[10, 10])
    plt.rcParams.update({'font.size': 18})
    sns.scatterplot(
        x=embedding[:, 1],
        y=embedding[:, 0],
        hue=txt_lab,
        alpha=0.2,
        ax=ax
    )
    plt.legend(frameon=False)
    #   plt.title('UMAP projections')
    cluster_index_list = []
    for a, anchor in enumerate(anchors):
        rect = patches.Rectangle(anchor, w, h, linewidth=3, edgecolor=colors[a], facecolor='none')
        ax.add_patch(rect)
        cluster_index = np.argwhere((anchor[0] < embedding[:, 1]) &
                                    ((anchor[0] + w) > embedding[:, 1]) &
                                    (anchor[1] < embedding[:, 0]) &
                                    ((anchor[1] + h) > embedding[:, 0])).flatten()
        cluster_index_list.append(cluster_index)
    plt.savefig('plots/UMAP/UMAP.svg')
    return cluster_index_list


def plot_profiles(pred, cluster_index, class_index=8, bin_size=1, color_edge='k', file_prefix='out'):
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    p = pred[cluster_index, :, class_index]
    ax.plot(np.repeat(p, bin_size, 1).T, alpha=0.4);
    # Set the borders to a given color...
    #     ax.tick_params(color=color_edge, labelcolor='green')


#   for spine in ax.spines.values():
#     spine.set_edgecolor(color_edge)
#     spine.set_linewidth(3)
#   plt.savefig('plots/UMAP/{}_coverage.svg'.format(file_prefix))


def plot_saliency_values(saliency_scores, X_sample, file_prefix='out'):
    fig, ax = plt.subplots(1, 1, figsize=[10, 6])
    #   fig = plt.figure(figsize=(20,2))
    for i in range(len(saliency_scores)):
        grad_times_input = np.sum(saliency_scores[i] * X_sample[i], axis=1)
        plt.plot(grad_times_input, alpha=0.4)


#   for spine in ax.spines.values():
#     spine.set_edgecolor(color_edge)
#     spine.set_linewidth(3)
#   plt.savefig('plots/UMAP/{}_{}_saliency.svg'.format(file_prefix, len(saliency_scores)))


def plot_saliency_logos(saliency_scores, X_sample, window=20, num_plot=25, titles=None, vline_pos=None):
    L = X_sample.shape[1]

    #   fig = plt.figure(figsize=(20,22))
    for i in range(len(saliency_scores)):
        #   for i in [6, 14, 58, 65, 78]:
        fig, ax = plt.subplots(1, 1, figsize=[20, 2])
        x_sample = np.expand_dims(X_sample[i], axis=0)
        scores = np.expand_dims(saliency_scores[i], axis=0)

        # find window to plot saliency maps (about max saliency value)
        index = np.argmax(np.max(np.abs(scores), axis=2), axis=1)[0]
        if index - window < 0:
            start = 0
            end = window * 2 + 1
        elif index + window > L:
            start = L - window * 2 - 1
            end = L
        else:
            start = index - window
            end = index + window

        saliency_df = tfomics.impress.grad_times_input_to_df(x_sample[:, start:end, :], scores[:, start:end, :])

        #     ax = plt.subplot(num_plot,1,i+1)
        tfomics.impress.plot_attribution_map(saliency_df, ax, figsize=(20, 1))
        if vline_pos:
            ax.axvline(vline_pos, linewidth=2, color='r')
        if titles:
            ax.set_title(titles[i])
    #     plt.savefig('plots/UMAP/saliency_plots/{}.svg'.format(i))
    plt.tight_layout()


def plot_saliency_logos_oneplot(saliency_scores, X_sample, window=20,
                                num_plot=25, titles=[], vline_pos=None,
                                filename=None):
    N, L, A = X_sample.shape
    fig, axs = plt.subplots(N, 1, figsize=[20, 2 * N])
    #   fig = plt.figure(figsize=(20,22))
    for i in range(N):
        ax = axs[i]
        x_sample = np.expand_dims(X_sample[i], axis=0)
        scores = np.expand_dims(saliency_scores[i], axis=0)

        # find window to plot saliency maps (about max saliency value)
        index = np.argmax(np.max(np.abs(scores), axis=2), axis=1)[0]
        if index - window < 0:
            start = 0
            end = window * 2 + 1
        elif index + window > L:
            start = L - window * 2 - 1
            end = L
        else:
            start = index - window
            end = index + window

        saliency_df = tfomics.impress.grad_times_input_to_df(x_sample[:, start:end, :], scores[:, start:end, :])

        #     ax = plt.subplot(num_plot,1,i+1)
        tfomics.impress.plot_attribution_map(saliency_df, ax, figsize=(20, 1))
        if vline_pos:
            ax.axvline(vline_pos, linewidth=2, color='r')
        if len(titles):
            ax.set_title(titles[i])
    #     plt.savefig('plots/UMAP/saliency_plots/{}.svg'.format(i))
    plt.tight_layout()
    if filename:
        assert not os.path.isfile(filename), 'File exists!'
        plt.savefig(filename, format='svg')


def get_multiple_saliency_values(seqs, model, class_index):
    explainer = Explainer(model, class_index=class_index)
    saliency_scores = explainer.saliency_maps(seqs)
    grad_times_input = np.sum(saliency_scores * seqs, axis=-1)
    return grad_times_input


def get_saliency_values(seq, model, class_index):
    explainer = explain.Explainer(model, class_index=class_index)
    x = np.expand_dims(seq, axis=0)
    saliency_scores = explainer.saliency_maps(x)
    grad_times_input = np.sum(saliency_scores[0] * seq, axis=1)
    return grad_times_input
