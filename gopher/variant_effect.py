import h5py
import numpy as np
import subprocess
import tensorflow as tf
import pandas as pd
import utils
from custom_fit import center_crop

def enforce_const_range(site, window):
    """
    Function to get constant size bed ranges
    :param site: center positions
    :param window: window size around center position
    :return: new starts and ends
    """
    half_window = np.round(window/2).astype(int)
    start = site - half_window
    end = site + half_window
    return start, end

def expand_range(bedfile, output_filename, window=3072):
    """
    Function to write a new bed file with expanded ranges
    :param bedfile: existing bed file, the ranges of which will be expanded
    :param output_filename: new bed file path
    :param window: window size
    :return: None
    """
    df = pd.read_csv(bedfile, sep='\t', header=None, index_col=None)
    start, end = enforce_const_range(df.iloc[:,1].astype(int), window)
    df_expanded = df.copy()
    df_expanded.iloc[:,1] = start.values
    df_expanded.iloc[:,2] = end.values
    df_nonneg = df_expanded[df_expanded.iloc[:,1]>0]
    df_nonneg = df_nonneg.reset_index(drop=True)
    df_nonneg.to_csv(output_filename, header=None, sep='\t', index=None)

def vcf_quantitative(model_path, ref_seq, alt_seq, input_size, output_pre, robust=True, batch_size=64, shift_num=10):
    """
    Wraper function for running variant effect prediction on quantitative models
    :param model_path: saving directory of model to be evaluated
    :param ref_seq: sequence input with reference allele in it
    :param alt_seq: sequence input with alternative allele in it
    :param input_size: input size of model
    :param output_pre: prefix for output files
    :param robust: Use robust prediction to mesaure effect size or not
    :param batch_size: batch size used during prediction
    :param shift_num: how many shifts to generate for robust predictions
    :return: a h5 file with effect size recorded for each ref/alt sequence pair
    """
    model = utils.read_model(model_path, False)[0]
    if robust == True:
        vcf_diff = vcf_robust(ref_seq, alt_seq, model, shift_num=shift_num,
                              window_size=input_size, batch_size=batch_size)

    elif robust == False:
        vcf_diff = vcf_fast(ref_seq, alt_seq, model,
                            window_size=input_size, batch_size=batch_size)
    else:
        raise ValueError('robust parameter only takes boolean values')

    vcf_diff = np.concatenate(vcf_diff)
    h5_output = h5py.File(output_pre + '.h5', 'w')
    h5_output.create_dataset('vcf_diff', data=vcf_diff)
    h5_output.close()


def vcf_binary(model_path, ref_seq, alt_seq, input_size, output_pre, robust=True, batch_size=64, shift_num=10):
    """
    Wraper function for running VCF on binary models
    :param model_path: saving directory of model to be evaluated
    :param ref_seq: sequence input with reference allele in it
    :param alt_seq: sequence input with alternative allele in it
    :param input_size: input size of model
    :param output_pre: prefix for output files
    :param robust: Use robust prediction to mesaure effect size or not
    :param batch_size: batch size used during prediction
    :param shift_num: how many shifts to generate for robust predictions
    :return: a h5 file with effect size recorded for each ref/alt sequence pair
    """
    model = utils.read_model(model_path, False)[0]
    if robust == True:
        vcf_diff = vcf_binary_robust(ref_seq, alt_seq, model, shift_num=shift_num,
                                     window_size=input_size, batch_size=batch_size)
    elif robust == False:
        vcf_diff = vcf_binary_fast(ref_seq, alt_seq, model,window_size = input_size,batch_size=batch_size)

    else:
        raise ValueError('robust parameter only takes boolean values')

    vcf_diff = np.concatenate(vcf_diff)
    h5_output = h5py.File(output_pre + '.h5', 'w')
    h5_output.create_dataset('vcf_diff', data=vcf_diff)
    h5_output.close()


def vcf_fast(ref, alt, model, window_size=2048, batch_size=64):
    """

    :param ref:
    :param alt:
    :param model:
    :param window_size:
    :param batch_size:
    :return:
    """
    if ref.shape[1] != window_size:
        ref,alt= center_crop(ref,alt,window_size)

    vcf_diff_list = []
    i = 0
    while i < len(ref):
        if i + batch_size < len(ref):
            ref_seq = ref[i:i + batch_size]
            alt_seq = alt[i:i + batch_size]
            batch_n = batch_size
            i = i + batch_size
        else:
            ref_seq = ref[i:len(ref)]
            alt_seq = alt[i:len(ref)]
            batch_n = len(ref) - i
            i = len(ref)

        ref_pred = model.predict(ref_seq)
        alt_pred = model.predict(alt_seq)

        bin_size = window_size / ref_pred.shape[1]
        ref_pred = np.repeat(ref_pred, bin_size, axis=1)
        alt_pred = np.repeat(alt_pred, bin_size, axis=1)

        ref_pred_1k = ref_pred[:, 512:1536, :]
        alt_pred_1k = alt_pred[:, 512:1536, :]

        vcf_diff = np.sum(alt_pred_1k, axis=1) / np.sum(ref_pred_1k, axis=1)
        vcf_diff_list.append(np.log2(vcf_diff))

    return vcf_diff_list


def vcf_robust(ref, alt, model, shift_num=10, window_size=2048, batch_size=64):
    """

    :param ref:
    :param alt:
    :param model:
    :param shift_num:
    :param window_size:
    :param batch_size:
    :return:
    """
    # calculate the coordinates for sequences to conserve in the center
    vcf_diff_list = []
    chop_size = ref.shape[1]
    center_idx = int(0.5 * (chop_size - window_size))
    center_range = np.array(range(center_idx, center_idx + window_size))
    conserve_size = window_size * 2 - chop_size
    conserve_start = chop_size // 2 - conserve_size // 2
    conserve_end = conserve_start + conserve_size - 1

    i = 0
    while i < len(ref):
        if i + batch_size < len(ref):
            ref_seq = ref[i:i + batch_size]
            alt_seq = alt[i:i + batch_size]
            batch_n = batch_size
            i = i + batch_size
        else:
            ref_seq = ref[i:len(ref)]
            alt_seq = alt[i:len(ref)]
            batch_n = len(ref) - i
            i = len(ref)

        # creat shifted sequence list and make predictions
        shifted_ref, shifted_alt, shift_idx = utils.window_shift(ref_seq, alt_seq,
                                                                window_size, shift_num, both_seq=True)
        ref_pred = model.predict(shifted_ref)
        alt_pred = model.predict(shifted_alt)
        bin_size = window_size / ref_pred.shape[1]
        ref_pred = np.repeat(ref_pred, bin_size, axis=1)
        alt_pred = np.repeat(alt_pred, bin_size, axis=1)

        # Select conserved part
        crop_start_i = conserve_start - shift_idx - center_idx
        crop_idx = crop_start_i[:, None] + np.arange(conserve_size)
        crop_idx = crop_idx.reshape(conserve_size * shift_num * batch_n)
        crop_row_idx = np.repeat(range(0, shift_num * batch_n), conserve_size)
        crop_f_index = np.vstack((crop_row_idx, crop_idx)).T.reshape(shift_num * batch_n, conserve_size, 2)

        # get pred 1k part
        ref_pred_1k = tf.gather_nd(ref_pred, crop_f_index)
        alt_pred_1k = tf.gather_nd(alt_pred, crop_f_index)

        sep_ref = np.array(np.array_split(ref_pred_1k, batch_n))
        sep_alt = np.array(np.array_split(alt_pred_1k, batch_n))

        # get average pred
        avg_ref = np.mean(sep_ref, axis=1)
        avg_alt = np.mean(sep_alt, axis=1)

        # get difference between average coverage value
        vcf_diff = np.sum(avg_alt, axis=1) / np.sum(avg_ref, axis=1)
        vcf_diff_list.append(np.log2(vcf_diff))

    return vcf_diff_list


def vcf_binary_fast(ref, alt, model, window_size,batch_size=64):
    """

    :param ref:
    :param alt:
    :param model:
    :param batch_size:
    :param layer:
    :return:
    """
    if ref.shape[1] != window_size:
        ref,alt= center_crop(ref,alt,window_size)

    vcf_diff_list = []
    i = 0
    while i < len(ref):
        if i + batch_size < len(ref):
            ref_seq = ref[i:i + batch_size]
            alt_seq = alt[i:i + batch_size]
            batch_n = batch_size
            i = i + batch_size
        else:
            ref_seq = ref[i:len(ref)]
            alt_seq = alt[i:len(ref)]
            batch_n = len(ref) - i
            i = len(ref)


        ref_pred = model.predict(ref_seq)
        alt_pred = model.predict(alt_seq)


        vcf_diff = alt_pred / ref_pred
        vcf_diff_list.append(np.log2(vcf_diff))

    return vcf_diff_list


def vcf_binary_robust(ref, alt, model, shift_num=10, window_size=2048, batch_size=64):
    """

    :param ref:
    :param alt:
    :param model:
    :param shift_num:
    :param window_size:
    :param batch_size:
    :param layer:
    :return:
    """
    # calculate the coordinates for sequences to conserve in the center
    vcf_diff_list = []
    chop_size = ref.shape[1]
    center_idx = int(0.5 * (chop_size - window_size))
    center_range = np.array(range(center_idx, center_idx + window_size))
    conserve_size = window_size * 2 - chop_size
    conserve_start = chop_size // 2 - conserve_size // 2
    conserve_end = conserve_start + conserve_size - 1

    i = 0
    while i < len(ref):
        if i + batch_size < len(ref):
            ref_seq = ref[i:i + batch_size]
            alt_seq = alt[i:i + batch_size]
            batch_n = batch_size
            i = i + batch_size
        else:
            ref_seq = ref[i:len(ref)]
            alt_seq = alt[i:len(ref)]
            batch_n = len(ref) - i
            i = len(ref)

        # creat shifted sequence list and make predictions
        shifted_ref, shifted_alt, shift_idx = utils.window_shift(ref_seq, alt_seq,
                                                                window_size, shift_num, both_seq=True)

        ref_pred = model.predict(shifted_ref)
        alt_pred = model.predict(shifted_alt)


        # seperate label per seq
        sep_ref = np.array(np.array_split(ref_pred, batch_n))
        sep_alt = np.array(np.array_split(alt_pred, batch_n))
        ##shape(batch_size,shift_num,15)

        average_ref = np.mean(sep_ref, axis=1)
        average_alt = np.mean(sep_alt, axis=1)
        ##shape(batch_size,15)

        # get difference between average coverage value
        vcf_diff = average_alt / average_ref
        vcf_diff_list.append(np.log2(vcf_diff))

    return vcf_diff_list


def dna_one_hot(seq):
    """
    Function to convert string DNA sequences into onehot
    :param seq: string DNA sequence
    :return: onehot sequence
    """
    seq_len = len(seq)
    seq_start = 0
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len, 4), dtype='float16')

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == 'A':
                seq_code[i, 0] = 1
            elif nt == 'C':
                seq_code[i, 1] = 1
            elif nt == 'G':
                seq_code[i, 2] = 1
            elif nt == 'T':
                seq_code[i, 3] = 1
            else:
                seq_code[i, :] = 0.25

    return seq_code


def convert_bed_to_seq(bedfile, output_fa, genomefile):
    """
    This function collects DNA sequences corresponding to a bedfile into a fasta file
    :param bedfile: existing bed file
    :param output_fa: new fasta path
    :param genomefile: genome fasta to use to get the sequences
    :return: list of coordinates and string sequences
    """
    cmd = 'bedtools getfasta -fi {} -bed {} -s -fo {}'.format(genomefile, bedfile, output_fa)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    _ = process.communicate()
    coords_list, seqs_list = fasta2list(output_fa)
    return coords_list, seqs_list


def fasta2list(fasta_file):
    """
    Function to convert fasta file to a list of DNA strings
    :param fasta_file: existing fasta file
    :return: list of coordinates and string sequences
    """
    fasta_coords = []
    seqs = []
    header = ''

    for line in open(fasta_file):
        if line[0] == '>':
            # header = line.split()[0][1:]
            fasta_coords.append(line[1:].rstrip())
        else:
            s = line.rstrip()
            s = s.upper()
            seqs.append(s)

    return fasta_coords, seqs
