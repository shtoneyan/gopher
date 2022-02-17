import tensorflow as tf
import filter_viz
import modelzoo
import util
import pandas as pd

# model_path = 'paper_runs/basenji/augmentation_basenji/run-20210924_160405-psjsjf84'
# first_conv_layer = 2
# model_name = 'basenji_128'
# model = modelzoo.load_model(model_path,compile = True)

model_path = 'paper_runs/bpnet/augmentation_48/run-20211006_190817-456uzbu4'
first_conv_layer = 1
model_name = 'bpnet_base'
model = modelzoo.load_model(model_path,compile = True)


# model_path = '/home/amber/QuantPred/paper_runs/binary/mod_binary/run-20211019_161849-9ygdmjss/files/best_model.h5'
# first_conv_layer = 3
# model_name = 'mod_binary'
# model = tf.keras.models.load_model(model_path)

# model_path = '/home/amber/QuantPred/paper_runs/binary/basset/run-20210825_040148-nieq47kf/files/best_model.h5'
# first_conv_layer = 2
# model_name = 'ori_basset'
# model = tf.keras.models.load_model(model_path)

#32 res relu/exp
#exp
# model_path = 'paper_runs/new_models/32_res/run-20211023_095136-to28llil'
# first_conv_layer = 3
# model_name = 'rb_32_exp'
#model = modelzoo.load_model(model_path,compile = True)

#relu
# model_path = 'paper_runs/new_models/32_res/run-20211023_095137-8shfejto'
# first_conv_layer = 3
# model_name = 'rb_32_relu'
#model = modelzoo.load_model(model_path,compile = True)

#base res relu/exp
#exp
# model_path = 'paper_runs/new_models/base_res/run-20211022_141032-m1cjyb3z'
# first_conv_layer = 3
# model_name = 'rb_base_exp'
#model = modelzoo.load_model(model_path,compile = True)
#relu
# model_path = 'paper_runs/new_models/base_res/run-20211101_111917-v4iozxug'
# first_conv_layer = 3
# model_name = 'rb_base_relu'
#model = modelzoo.load_model(model_path,compile = True)








output_dir = 'datasets/tomtom/'
profile_data_dir = '/home/shush/profile/QuantPred/datasets/complete/peak_centered/i_2048_w_1'
testset = util.make_dataset(profile_data_dir, 'test', util.load_stats(profile_data_dir), batch_size=128,shuffle = False)
test_x = testset.map(lambda x,y: x)

max_filter,counter = filter_viz.filter_max_align_batch(test_x,model,layer = first_conv_layer)
clip_filter = filter_viz.clip_filters(max_filter, threshold=0.5, pad=3)
filter_viz.meme_generate(clip_filter,output_file = output_dir+model_name+'.txt')
