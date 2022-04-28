import os
import sys
sys.path.append('../gopher')
import saliency_embed, utils
import glob
import tensorflow as tf

cell_line = 13  # cell line index
umap_dir = utils.make_dir('inter_results/umap_embeddings')
run_path = glob.glob('../trained_models/**/run-20211023_095131-w6okxt01', recursive=True)[0]
out_dir = utils.make_dir(os.path.join(umap_dir, os.path.basename(run_path)))
# get test set
testset, targets = utils.collect_whole_testset('../datasets/quantitative_data/testset/', coords=True)
np_C, np_X, np_Y = utils.convert_tfr_to_np(testset)

# load and get model layer
layer = -3
model, bin_size = utils.read_model(run_path, compile_model=False)
aux_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

for cell_line in range(len(targets)):
    csv_filepath = os.path.join(out_dir, str(cell_line) + '_UMAP_embeddings.csv')
    thresholded_C, thresholded_X, thresholded_Y = utils.threshold_cell_line_np(np_C,
                                                                               np_X,
                                                                               np_Y,
                                                                               cell_line,
                                                                               more_than=2)

    # get IDR peaks from cell line specific test set
    idr_class = saliency_embed.label_idr_peaks(thresholded_C, cell_line,
                                               bedfile1='../datasets/quantitative_data/testset/sequences.bed',
                                               bedfile2='../datasets/quantitative_data/cell_line_testsets/cell_line_{}/complete/peak_centered/i_2048_w_1.bed'.format(
                                                   cell_line),
                                               fraction_overlap=0.5)
    # get penultimate representations
    interm_representations = utils.predict_np(thresholded_X, aux_model, batch_size=32, reshape_to_2D=True)
    # embed in UMAP
    embeddings = saliency_embed.get_embeddings(interm_representations)
    embeddings['IDR'] = idr_class
    embeddings['cell line'] = targets[cell_line]
    embeddings.to_csv(csv_filepath, index=None)
