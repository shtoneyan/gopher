import numpy as np
import util
import tensorflow as tf
import subprocess


def vcf_fast(ref, alt, model, window_size=2048, batch_size=64):
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
        shifted_ref, shifted_alt, shift_idx = util.window_shift(ref_seq, alt_seq,
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


def vcf_binary_fast(ref, alt, model, batch_size=64, layer=-1, diff_func='effect_size'):
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

def vcf_binary_robust(ref, alt, model, shift_num=10, window_size=2048, batch_size=64, layer=-1, diff_func='effect_size'):
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
        shifted_ref, shifted_alt, shift_idx = util.window_shift(ref_seq, alt_seq,
                                                                window_size, shift_num, both_seq=True)
        if int(layer) == -1:
            ref_pred = model.predict(shifted_ref)
            alt_pred = model.predict(shifted_alt)
        elif int(layer) == -2:
            intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                                      outputs=model.output.op.inputs[0].op.inputs[0])
            ref_pred = intermediate_layer_model.predict(shifted_ref)
            alt_pred = intermediate_layer_model.predict(shifted_alt)

        # seperate label per seq
        sep_ref = np.array(np.array_split(ref_pred, batch_n))
        sep_alt = np.array(np.array_split(alt_pred, batch_n))
        ##shape(batch_size,shift_num,15)

        average_ref = np.mean(sep_ref, axis=1)
        average_alt = np.mean(sep_alt, axis=1)
        ##shape(batch_size,15)

        # get difference between average coverage value
        if diff_func == 'effect_size':
            vcf_diff = average_alt / average_ref
            vcf_diff_list.append(np.log2(vcf_diff))
        elif diff_func == 'log_ratio':
            vcf_diff = np.log2(average_alt) / np.log2(average_ref)
            vcf_diff_list.append(vcf_diff)

    return vcf_diff_list


def dna_one_hot(seq):
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


def convert_bed_to_seq(bedfile, output_fa, genomefile='/home/shush/genomes/hg38.fa'):
    cmd = 'bedtools getfasta -fi {} -bed {} -s -fo {}'.format(genomefile, bedfile, output_fa)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    coords_list, seqs_list = fasta2list(output_fa)
    return coords_list, seqs_list



def fasta2list(fasta_file):
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