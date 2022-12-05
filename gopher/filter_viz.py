import numpy as np
from tensorflow import keras

def filter_max_align_batch(X, model, layer=3, window=24, threshold=0.5, batch_size=1024, max_align=1e4, verbose=1):
  """get alignment of filter activations for visualization"""
  if verbose:
    print("Calculating filter PPM based on activation-based alignments")
  N,L,A = X.element_spec.shape
  num_filters = model.layers[layer].output.shape[2]

  # Set the left and right window sizes
  window_left = int(window/2)
  window_right = window - window_left

  # get feature maps of 1st convolutional layer after activation
  intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

  #dataset = tf.data.Dataset.from_tensor_slices(X)
  #batches = X.batch(batch_size)
  batches = X
  # loop over batches to capture MAX activation
  if verbose:
    print('  Calculating MAX activation')
  MAX = np.zeros(num_filters)
  for x in batches:

    # get feature map for mini-batch
    fmap = intermediate.predict(x)

    # loop over each filter to find "active" positions
    for f in range(num_filters):
      MAX[f] = np.maximum(MAX[f], np.max(fmap[:,:,f]))


  # loop over each filter to find "active" positions

  W = []
  counts = []
  for f in range(num_filters):
    if verbose:
      print("    processing %d out of %d filters"%(f+1, num_filters))
    status = 0

    # compile sub-model to get feature map
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output[:,:,f])

    # loop over each batch
    #dataset = tf.data.Dataset.from_tensor_slices(X)
    seq_align_sum = np.zeros((window, A)) # running sum
    counter = 0                            # counts the number of sequences in alignment
    status = 1                            # monitors whether depth of alignment has reached max_align
    for x in X:
      if status:

        # get feature map for a batch sequences
        fmaps = intermediate.predict(x)

        # Find regions above threshold
        for data_index, fmap in enumerate(fmaps):
          if status:
            pos_index = np.where(fmap > MAX[f] * threshold)[0]

            # Make a sequence alignment centered about each activation (above threshold)
            for i in range(len(pos_index)):
              if status:
                # Determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # Check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                  seq_align_sum += x[data_index,start_window:end_window,:].numpy()
                  counter += 1
                  if counter > max_align:
                    status = 0
                else:
                  break
          else:
            break
      else:
        if verbose:
          print("      alignment has reached max depth for all filters")
        break

    # calculate position probability matrix of filter
    if verbose:
      print("      %d sub-sequences above threshold"%(counter))
    if counter > 0:
      W.append(seq_align_sum/counter)
    else:
      W.append(np.ones((window,A))/A)
    counts.append(counter)
  return np.array(W), np.array(counts)


def clip_filters(W, threshold=0.5, pad=3):
  """clip uninformative parts of conv filters"""
  W_clipped = []
  for w in W:
    L,A = w.shape
    entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=1)
    index = np.where(entropy > threshold)[0]
    if index.any():
      start = np.maximum(np.min(index)-pad, 0)
      end = np.minimum(np.max(index)+pad+1, L)
      W_clipped.append(w[start:end,:])
    else:
      W_clipped.append(w)

  return W_clipped


def meme_generate(W, output_file='meme.txt', prefix='filter'):
  """generate a meme file for a set of filters, W ∈ (N,L,A)"""

  # background frequency
  nt_freqs = [1./4 for i in range(4)]

  # open file for writing
  f = open(output_file, 'w')

  # print intro material
  f.write('MEME version 4\n')
  f.write('\n')
  f.write('ALPHABET= ACGT\n')
  f.write('\n')
  f.write('Background letter frequencies:\n')
  f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
  f.write('\n')

  for j, pwm in enumerate(W):
    L, A = pwm.shape
    f.write('MOTIF %s%d \n' % (prefix, j))
    f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (L, L))
    for i in range(L):
      f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[i,:]))
    f.write('\n')

  f.close()


def count_meme_entries(motif_path):
  """Count number of meme entries"""
  with open(motif_path, 'r') as f:
    counter = 0
    for line in f:
      if line[:6] == 'letter':
        counter += 1
  return counter
