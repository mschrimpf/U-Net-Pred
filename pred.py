import os
import pdb
import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from tensorflow.keras import backend as K
from model.model_fn import build_compile_model
from tensorflow.keras import applications

parser = argparse.ArgumentParser(
    description='gamma-net predicts log10 gamma power for a given image')
parser.add_argument('--input', type=str, nargs=1,
                    help='input size 84x84 .png or .npy', default='examples/sample.png')
parser.add_argument('--output', type=str, nargs=1,
                    help='output predictability type "structure" or "content"', default='structure')
args = parser.parse_args()
input_name = args.input[0]
flag_LPIPS = args.output[0]=='content'

# params
params = {'weight_dir': 'weights.last.h5', 'lr': 1.56*1e-4,
          'p_content': 0, 'p_style': 0, 'p_hole': 0, 'p_tv': 0,
          'p_fft_s': 0.002, 'p_fft_c': 8.79,  # 'p_fft_s': 0.330, 'p_fft_c': 505.47,
          'p_fft_abs': 1, 'p_fft_log': 0.026, 'p_fft_phase': 6.45e-05,  # .0556
          'b_size': 4, 'current_epoch': 0,
          'n_epoch': 150000,
          'exp_dir': 'experiments/',
          'inst_norm': False,
          'input_size': 224
          }

WEIGHT_DIR = 'weights.last.h5'
BATCH_SIZE = 64
PXD = 26 # pixels per degree - should be even 

# input
out_name, file_ext = os.path.splitext(input_name)

# model
model = build_compile_model(params)
model.summary()

# modelname
flag_numpy = 1 if file_ext=='.npy' else 0

if flag_numpy:
    this_input = np.load(input_name)
    test_steps = 1
else:
    from skimage.io import imread
    from skimage.transform import resize
    img = imread(input_name)
    this_input = np.expand_dims(img, axis=0)
    test_steps = 1
    
#
this_input = this_input.astype(np.float32)
NO_INPUT = this_input.shape[0]
HALF_SIZE = int(this_input.shape[1]/2)

# make simple mask
im_mask = np.ones(this_input.shape)
im_mask[:, HALF_SIZE-PXD:HALF_SIZE+PXD, HALF_SIZE-PXD:HALF_SIZE+PXD, :] = 0

# VGG-16 preprocessing
for ii in range(NO_INPUT):
    this_input[ii, :, :, :] = this_input[ii, :, :, ::-1]
    this_input[ii, :, :, :] -= [103.939, 116.779, 123.68]
pred = model.predict([this_input]+[im_mask], steps=test_steps)

# post process
for ii in range(NO_INPUT):
    img = pred[ii, :, :, :]
    img += [103.939, 116.779, 123.68]
    this_input[ii, :, :, :] += [103.939, 116.779, 123.68]
    img[img < 0] = 0
    img[img > 255] = 255
    img = img[:,:,::-1]
    this_input[ii, :, :, :] = this_input[ii,:,:,::-1]

# structure predictability
predictability = np.zeros((NO_INPUT, ))
if not flag_LPIPS:
    for ii in range(NO_INPUT):
        predictability[ii] = pearsonr(pred[ii, HALF_SIZE-PXD:HALF_SIZE+PXD, HALF_SIZE-PXD:HALF_SIZE+PXD,:].flatten(), 
                                    this_input[ii, HALF_SIZE-PXD:HALF_SIZE+PXD, HALF_SIZE-PXD:HALF_SIZE+PXD,:].flatten())[0]
else: # LPIPS predictability
    from LPIPS_VGG import get_LPIPS
    
    predictability = np.mean(get_LPIPS(pred[:, HALF_SIZE-PXD:HALF_SIZE+PXD, HALF_SIZE-PXD:HALF_SIZE+PXD,:],
                                         this_input[:, HALF_SIZE-PXD:HALF_SIZE+PXD, HALF_SIZE-PXD:HALF_SIZE+PXD,:]), axis=0)

# output
if flag_numpy:
    print('Predictability', predictability)
    np.save(out_name + '_predictability.npy', predictability)
else:
    print('Predictability', predictability)
