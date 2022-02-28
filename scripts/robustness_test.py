


def get_center_coordinates(coord, conserve_start, conserve_end):
    """

    :param coord:
    :param conserve_start:
    :param conserve_end:
    :return:
    """
    '''Extract coordinates according to robsutness test procedure'''
    chrom, start, end = str(np.array(coord)).strip('\"b\'').split('_')
    start, end = np.arange(int(start), int(end))[[conserve_start, conserve_end]]
    return (chrom, start, end)





def batch_pred_robustness_test(testset, sts, model,
                               shift_num=20, window_size=2048, get_preds=True):
    """
    This function obtains a dictionary of robust predictions (predictions averaged across multiple shifts), center
    predictions and variance based on robustness test for a given test set, center 1K coordinates and center 1K ground
    truth values. During robustness test we shift a 2K window within a given input 3K sequence multiple times,
    get predictions and calculate the center 1K (overlap region) average predictions as 'robust' prediction and
    variance.
    :param testset: testset in tf dataset format
    :param sts: statistics file of the dataset
    :param model: loaded trained model
    :param shift_num: number of times to shift
    :param window_size:
    :param get_preds:
    :return:
    """
    predictions_and_variance = {}
    predictions_and_variance['var'] = []  # prediction variance
    predictions_and_variance['robust_pred'] = []  # robust predictions
    predictions_and_variance['center_pred'] = []  # 1 shot center predictions
    center_1K_coordinates = []  # coordinates of center 1K
    center_ground_truth_1K = []  # center 1K ground truth base res

    chop_size = sts['target_length']
    center_idx = int(0.5 * (chop_size - window_size))
    conserve_size = window_size * 2 - chop_size
    conserve_start = chop_size // 2 - conserve_size // 2
    conserve_end = conserve_start + conserve_size - 1
    flanking_size = conserve_size // 2

    for t, (C, seq, Y) in enumerate(testset):
        batch_n = seq.shape[0]
        shifted_seq, _, shift_idx = util.window_shift(seq, seq, window_size, shift_num)

        # get prediction for shifted read
        shift_pred = model.predict(shifted_seq)
        bin_size = window_size / shift_pred.shape[1]
        shift_pred = np.repeat(shift_pred, bin_size, axis=1)

        # Select conserve part only
        crop_start_i = conserve_start - shift_idx - center_idx
        crop_idx = crop_start_i[:, None] + np.arange(conserve_size)
        crop_idx = crop_idx.reshape(conserve_size * shift_num * batch_n)
        crop_row_idx = np.repeat(range(0, shift_num * batch_n), conserve_size)
        crop_f_index = np.vstack((crop_row_idx, crop_idx)).T.reshape(shift_num * batch_n, conserve_size, 2)

        # get pred 1k part
        shift_pred_1k = tf.gather_nd(shift_pred[range(shift_pred.shape[0]), :, :], crop_f_index)
        sep_pred = np.array(np.array_split(shift_pred_1k, batch_n))
        var_pred = np.var(sep_pred, axis=1)
        predictions_and_variance['var'].append(np.sum(var_pred, axis=1))
        if get_preds:
            predictions_and_variance['robust_pred'].append(sep_pred.mean(axis=1))

            # get prediction for center crop
            center_2K_seq = seq[:, conserve_start - flanking_size:conserve_end + 1 + flanking_size, :]
            center_2K_pred = model.predict(center_2K_seq)
            center_2K_pred_unbinned = np.repeat(center_2K_pred, window_size // center_2K_pred.shape[1], axis=1)
            predictions_and_variance['center_pred'].append(
                center_2K_pred_unbinned[:, (conserve_size - flanking_size):(conserve_size + flanking_size), :])

            # add ground truth center 1K coordinates and coverage
            center_1K_coordinates.append([get_center_coordinates(coord, conserve_start, conserve_end) for coord in C])
            center_ground_truth_1K.append(Y.numpy()[:, conserve_start:conserve_end + 1, :])
    return predictions_and_variance, center_1K_coordinates, center_ground_truth_1K

