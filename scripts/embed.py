import tfr_evaluate
import subprocess
import numpy as np
import pandas as pd
import utils
import umap.umap_ as umap

def select(embeddings, lower_lim_1=None,
           upper_lim_1=None, lower_lim_2=None,
           upper_lim_2=None, idr=''):
    mask = np.zeros((embeddings['UMAP 1'].shape[0]))+1
    if lower_lim_1:
        mask *= (embeddings['UMAP 1']>lower_lim_1).values
    if upper_lim_1:
        mask *= (embeddings['UMAP 1']<upper_lim_1).values
    if lower_lim_2:
        mask *= (embeddings['UMAP 2']>lower_lim_2).values
    if upper_lim_2:
        mask *= (embeddings['UMAP 2']<upper_lim_2).values
    if idr=='y':
        print('Choosing only IDR')
        mask *= (embeddings['IDR']==True).values
    if idr=='n':
        print('Choosing only non IDR')
        mask *= (embeddings['IDR']!=True).values
    return mask.astype(bool)

# def describe_run(run_path, columns_of_interest=['model_fn', 'bin_size', 'crop', 'rev_comp']):
#     metadata = tfr_evaluate.get_run_metadata(run_path)
#     if 'data_dir' in metadata.columns:
#         model_id = [metadata['data_dir'].values[0].split('/i_3072_w_1')[0].split('/')[-1]]
#     else:
#         model_id = []
#     for c in columns_of_interest:
#         if c in metadata.columns:
#             model_id.append(str(metadata[c].values[0]))
#     return ' '.join(model_id)

def get_cell_line_overlaps(cell_line):
    cmd = 'bedtools intersect -f 0.5 -wa -a /home/shush/profile/QuantPred/datasets/chr8/complete/random_chop/i_2048_w_1/sequences.bed -b /home/shush/profile/QuantPred/datasets/cell_line_specific_test_sets/cell_line_{}/complete/peak_centered/i_2048_w_1.bed | uniq > chr8_cell_line_{}_IDR.bed'.format(cell_line, cell_line)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    df = pd.read_csv('chr8_cell_line_{}_IDR.bed'.format(cell_line), sep='\t', header=None)
    idr_starts = df.iloc[:,1].values
    return idr_starts

def threshold_cell_line_testset(testset, cell_line, more_than=2, less_than=None):
    np_C, np_X, np_Y = util.convert_tfr_to_np(testset, 3)
    return threshold_cell_line_np(np_C, np_X, np_Y, cell_line, more_than, less_than)

def threshold_cell_line_np(np_C, np_X, np_Y, cell_line, more_than, less_than=None):
    m1 = np_Y[:,:,cell_line].max(axis=1)>more_than
    if less_than:
        m2 = np_Y[:,:,cell_line].max(axis=1)<less_than
        threshold_mask = (m1&m2)
    else:
        threshold_mask = m1
    thresholded_X = np_X[threshold_mask]
    thresholded_C = np_C[threshold_mask]
    thresholded_Y = np_Y[threshold_mask,:,cell_line]
    return (thresholded_C, thresholded_X, thresholded_Y)

def label_idr_peaks(thresholded_C, cell_line):
    idr_class = []
    idr_starts = get_cell_line_overlaps(cell_line)
    idr_class.append([True if int(str(c).strip('\'b').split('_')[1]) in idr_starts else False for c in thresholded_C])
    idr_class = [item for sublist in idr_class for item in sublist]
    return idr_class

def predict_np(X, model, batch_size=32, reshape_to_2D=False):
    model_output = []
    for x_batch in util.batch_np(X, batch_size):
        model_output.append(model(x_batch).numpy())
    model_output = np.squeeze(np.concatenate(model_output))
    if reshape_to_2D:
        assert len(model_output.shape)==3, 'Wrong dimension for reshape'
        d1, d2, d3 = model_output.shape
        model_output = model_output.reshape(d1, d2*d3)
    return model_output

def get_embeddings(input_features):
    reducer = umap.UMAP(random_state=28)
    embedding = reducer.fit_transform(input_features)
    df = pd.DataFrame({'UMAP 1':embedding[:,1], 'UMAP 2':embedding[:,0]}) #, 'IDR':idr_class})
    return df

def get_GC_content(seq_array):
    return 2*seq_array[:,:,1].sum(axis=-1)/seq_array.shape[1]

def add_motif_to_seq(motif_tuple_list, src_seq, dst_seq):
    all_patterns = ''
    for motif_tuple in motif_tuple_list:
        motif_start, motif_pattern = motif_tuple
        onehot_motif = src_seq.copy()[motif_start:motif_start+len(motif_pattern)]
        dst_seq_copy = dst_seq.copy()
        dst_seq_copy[motif_start:motif_start+len(motif_pattern)] = onehot_motif
        all_patterns += (motif_pattern+', ')
    all_patterns = all_patterns.rstrip(', ')
    return dst_seq_copy, all_patterns


def plot_mean_coverages(data_and_labels, ax):
    for i, (data, label, p) in enumerate(data_and_labels):
        if 'non' in label:
            marker_style = '--'
        else:
            marker_style = '-'
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = (est - sd, est + sd)
        ax.fill_between(x,cis[0], cis[1], alpha=0.08, color=p)
        ax.plot(x,  est,p,label=label)
