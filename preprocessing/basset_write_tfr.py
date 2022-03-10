import h5py
import sys
from collections import OrderedDict
import numpy as np

def main():
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    h5f = h5py.File(output_file, "w")
    for dataset in ['train','valid','test']:
        fasta_file = input_dir+dataset+'.fa'
        targets_file = input_dir + dataset+'_actfile.bed'
        print("LOADING DATA")
        seqs, targets = load_data_1hot(
            fasta_file,
            targets_file,
            extend_len=None,
            mean_norm=False,
            whiten=False,
            permute=False,
            sort=False,
        )

        seqs = seqs.reshape((seqs.shape[0], int(seqs.shape[1] / 4), 4))
        
        h5f.create_dataset("x_"+dataset, data=seqs)
        h5f.create_dataset("y_"+dataset, data=targets)
        
    h5f.close()

#######################################################################################################
def align_seqs_scores_1hot(seq_vecs, seq_scores, sort=True):
    if sort:
        seq_headers = sorted(seq_vecs.keys())
    else:
        seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_scores = []
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])
        train_scores.append(seq_scores[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)
    train_scores = np.vstack(train_scores)

    return train_seqs, train_scores

def dna_one_hot(seq, seq_len=None, flatten=True, n_random=False):
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim : seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_random:
        seq_code = np.zeros((seq_len, 4), dtype="bool")
    else:
        seq_code = np.zeros((seq_len, 4), dtype="float16")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i, 0] = 1
            elif nt == "C":
                seq_code[i, 1] = 1
            elif nt == "G":
                seq_code[i, 2] = 1
            elif nt == "T":
                seq_code[i, 3] = 1
            else:
                if n_random:
                    ni = random.randint(0, 3)
                    seq_code[i, ni] = 1
                else:
                    seq_code[i, :] = 0.25

    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None, :]

    return seq_vec

def hash_sequences_1hot(fasta_file, extend_len=None):
    # determine longest sequence
    if extend_len is not None:
        seq_len = extend_len
    else:
        seq_len = 0
        seq = ""
        for line in open(fasta_file):
            if line[0] == ">":
                if seq:
                    seq_len = max(seq_len, len(seq))

                header = line[1:].rstrip()
                seq = ""
            else:
                seq += line.rstrip()

        if seq:
            seq_len = max(seq_len, len(seq))

    # load and code sequences
    seq_vecs = OrderedDict()
    seq = ""
    for line in open(fasta_file):
        if line[0] == ">":
            if seq:
                seq_vecs[header] = dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ""
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = dna_one_hot(seq, seq_len)

    return seq_vecs

def hash_scores(scores_file):
    seq_scores = {}

    for line in open(scores_file):
        a = line.split()

        try:
            loci_key = a[0]+':'+ a[1]+'-'+a[2]+'('+a[3]+')'
            seq_scores[loci_key] = np.array([float(a[i]) for i in range(4, len(a))])
        except Exception:
            print("Ignoring header line", file=sys.stderr)

    # consider converting the scores to integers
    int_scores = True
    for header in seq_scores:
        if not np.equal(np.mod(seq_scores[header], 1), 0).all():
            int_scores = False
            break

    if int_scores:
        for header in seq_scores:
            seq_scores[header] = seq_scores[header].astype("int8")

        """
        for header in seq_scores:
            if seq_scores[header] > 0:
                seq_scores[header] = np.array([0, 1], dtype=np.min_scalar_type(1))
            else:
                seq_scores[header] = np.array([1, 0], dtype=np.min_scalar_type(1))
        """

    return seq_scores

def load_data_1hot(
    fasta_file,
    scores_file,
    extend_len=None,
    mean_norm=True,
    whiten=False,
    permute=True,
    sort=False,
):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file, extend_len)
    
    # load scores
    seq_scores = hash_scores(scores_file)

    # align and construct input matrix
    train_seqs, train_scores = align_seqs_scores_1hot(seq_vecs, seq_scores, sort)

    # whiten scores
    if whiten:
        train_scores = preprocessing.scale(train_scores)
    elif mean_norm:
        train_scores -= np.mean(train_scores, axis=0)

    # randomly permute
    if permute:
        order = npr.permutation(train_seqs.shape[0])
        train_seqs = train_seqs[order]
        train_scores = train_scores[order]

    return train_seqs, train_scores


if __name__ == '__main__':
  main()