import glob
import os
import sys
sys.path.append('../gopher')
import utils
import tensorflow as tf
from modelzoo import GELU
from tensorflow.python.framework.ops import disable_eager_execution


# save binary model logits for basenji binary model
binary_basenji_path = '../trained_models/binary/basenji_binary/run-20220404_153248-gopd8diq'
model = utils.read_model(binary_basenji_path)[0]
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                      outputs=model.get_layer('dense').output)
intermediate_layer_model.save('temp.h5')
disable_eager_execution()
temp_model = tf.keras.models.load_model('temp.h5'
                                        ,custom_objects={'GELU':GELU})
intermediate_layer_model = tf.keras.Model(inputs=temp_model.input,
                                                      outputs=temp_model.output.op.inputs[0])
intermediate_layer_model.save(os.path.join(binary_basenji_path, 'logit.h5'))
os.remove('temp.h5')
# save logits for other models
run_paths = [r for r in glob.glob('../trained_models/binary/*/run*') if 'basenji' not in r]
for run_path in run_paths:
    print(run_path)
    model, _ = utils.read_model(run_path)
    if 'activation' in model.layers[-1].name:
        model_logits = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    else:
        model_logits = tf.keras.Model(inputs=model.input,
                               outputs=model.output.op.inputs[0].op.inputs[0])
    model_logits.save(os.path.join(run_path, 'logit.h5'))
