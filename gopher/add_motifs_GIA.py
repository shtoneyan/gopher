import tensorflow as tf
import h5py, os, yaml
import numpy as np
import pandas as pd
from modelzoo import GELU
from tqdm import tqdm
import utils
import explain
import embed
import metrics
import quant_GIA
from optparse import OptionParser

def main():
    usage = 'usage: %prog [options] <run_path> <motif> <background_model> <cell_lines>'
    parser = OptionParser(usage)

    parser.add_option('-o', dest='out_dir',
        default='seamonster/add_GIA_interaction',
        help='Output directory [Default: %default]')
    parser.add_option('-n','--n_background', dest='n_background',
        default=1000, type='int',
        help='Sample number for background [Default: %default]')
    parser.add_option('--logits', dest='logits',
        default=False, action='store_true',
        help='Save umap array into TFRecords [Default: %default]')
    (options, args) = parser.parse_args()
    if len(args) != 4:
      parser.error('Must provide run path, motifs, background model and cell line.')
    else:
      run_path = args[0]
      motif_cluster = args[1].split(',')
      background_model = args[2]
      cell_lines = args[3].split(',')


    utils.make_dir(options.out_dir)
    testset, targets = utils.collect_whole_testset(coords=True)
    model = tf.keras.models.load_model(run_path, custom_objects={"GELU": GELU})
    if options.logits:
        print('USING LOGITS!')
        model = tf.keras.Model(inputs=model.input,
                                  outputs=model.output.op.inputs[0].op.inputs[0])
        suffix = 'logits_'
    else:
        suffix = ''

    run_name = suffix + [p for p in run_path.split('/') if 'run-' in p][0]
    gia_add_dir = utils.make_dir(os.path.join(options.out_dir, run_name))
    C, X, Y = utils.convert_tfr_to_np(testset, 3)

    if background_model == 'dinuc':
        X_set = quant_GIA.select_set('all_threshold', C, X, Y)
    elif background_type == 'none':
        X_set = quant_GIA.select_set('cell_low', C, X, Y)

    gi = quant_GIA.GlobalImportance(model, targets)
    gi.set_null_model(background_model, base_sequence=X_set, num_sample=options.n_background)
    for cell_line_name in cell_lines:
        optimized_motifs = []
        for motif in motif_cluster:
            print('Optimizing motif '+motif)
            base_dir = utils.make_dir(os.path.join(gia_add_dir, '{}_{}'.format(cell_line_name, motif)))
            output_dir = utils.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, options.n_background)))
            flanks_path = os.path.join(output_dir, 'flanks.csv')
            optimized_motifs.append(record_flank_test(gi, motif, targets,
                                                     cell_line_name, flanks_path))
        if len(motif_cluster)==2:
            print('Testing distance effect on motif interaction')
            # check for positional interaction by fixing one in the middle and sliding the other
            base_dir = utils.make_dir(os.path.join(gia_add_dir, '{}_{}_and_{}'.format(cell_line_name, motif_cluster[0], motif_cluster[1])))
            output_dir = utils.make_dir(os.path.join(base_dir, '{}_N{}'.format(background_model, options.n_background)))
            # test_interaction(gi, optimized_motifs, targets, output_dir, 'optimal_position_interaction.csv')
            # for first_motif_pos in [800, 1024, 1200]:
            for first_motif_pos in [1024]:
                print(first_motif_pos)
                distance_path = os.path.join(output_dir, str(first_motif_pos)+'_distance.csv')
                df = optimize_distance(gi, optimized_motifs, targets, distance_path, first_motif_pos)
                best_position = int(df[df['cell line']==cell_line_name].sort_values('mean difference').iloc[-1,0].split('_')[1])
                motif_pos_pairs = [(optimized_motifs[0], first_motif_pos),
                                    (optimized_motifs[1], best_position)]
                test_interaction(gi, motif_pos_pairs, targets, output_dir, str(first_motif_pos)+'_best_distance_interaction.csv')

def test_interaction(gi, optimized_motifs, targets, output_dir, filename):
    # filenames = ['best_distance_interaction.csv', 'optimal_position_interaction.csv']

    motifs_to_test = [ [(optimized_motifs[0][0], optimized_motifs[0][1])],
                            [(optimized_motifs[1][0], optimized_motifs[1][1])],
                            [(optimized_motifs[0][0], optimized_motifs[0][1]), (optimized_motifs[1][0], optimized_motifs[1][1])] ]
    interaction_test_dfs = []
    for motif_to_test in motifs_to_test:
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m,p in motif_to_test])
        diff = gi.embed_predict_quant_effect(motif_to_test).mean(axis=1)
        df = pd.DataFrame({
                            'mean difference':np.array(diff).flatten(),
                            'cell line': np.tile(targets, diff.shape[0])})
        df['motif'] = pattern_label
        interaction_test_dfs.append(df)
    pd.concat(interaction_test_dfs).to_csv(os.path.join(output_dir, filename))

def optimize_distance(gi, optimized_motifs, targets, distance_path, first_motif_pos):
    if os.path.isfile(distance_path):
        df = pd.read_csv(distance_path)
    else:
        two_motifs_pos_scores = []
        positions = list(range(0,2048-len(optimized_motifs[1]),2))
        for position in tqdm(positions):
            diff_scores = gi.embed_predict_quant_effect([(optimized_motifs[0], first_motif_pos),
                                                        (optimized_motifs[1],
                                                            position)])
            two_motifs_pos_scores.append(diff_scores.mean(axis=0).mean(axis=0))
        two_motifs_pos_scores = np.array(two_motifs_pos_scores)
        motif_2_label = np.array(['{}_{}'.format(optimized_motifs[1], pos) for pos in positions])
        df = pd.DataFrame({
                            'motif 2':np.repeat(motif_2_label, len(targets)),
                            'mean difference':np.array(two_motifs_pos_scores).flatten(),
                            'cell line': np.tile(targets, two_motifs_pos_scores.shape[0])})
        df['motif 1'] = '{}_{}'.format(optimized_motifs[0], first_motif_pos)
        df.to_csv(distance_path, index=None)
    return df


def record_flank_test(gi, motif, targets, cell_line_name, flanks_path):
    # select the best flanks based on where the dots are in the pattern
    if '.' in motif:
        if os.path.isfile(flanks_path):
            flank_scores = pd.read_csv(flanks_path)
        else:
            all_motifs = quant_GIA.generate_flanks(motif)
            print('Testing flanks')
            flank_scores = quant_GIA.test_flanks(gi, all_motifs, targets,
                                 output_path=flanks_path)
        best_flank = flank_scores[flank_scores['cell line'] == cell_line_name].sort_values('mean difference').iloc[-1,0]
        print(best_flank)
    else:
        best_flank = motif
    return best_flank

################################################################################
if __name__ == '__main__':
  main()
