from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import numpy as np
import argparse

def get_vgg(nrows, ncols, out_layer=None, is_trainable=False, h=True):
    # make vgg percept up to pool 3
    vgg_model = VGG16(weights="imagenet",
                        include_top=True,
                        input_shape=(nrows, ncols, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    outputs = [layer_dict[out].output for out in layer_dict.keys()]
    outputs = outputs[1:]
    
    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = False
    # model.compile(loss='mse', optimizer='adam')
    return model

# arguments
parser = argparse.ArgumentParser(
    description='gamma-net predicts log10 gamma power for a given image')
parser.add_argument('--input', type=str, nargs=1,
                    help='input size 84x84 .png or .npy', default='examples/sample.png')
args = parser.parse_args()
input_name = args.input[0]

# load data
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
this_input = this_input.astype(np.float32)

# preprocessing
this_input = this_input[:,:,:,::-1]
for ii in range(im_batch.shape[0]):
    this_input[ii, :, :, :] -= [103.939, 116.779, 123.68]

# VGG16
nrows, ncols = this_input.shape[1:3]
vgg_model = get_vgg(nrows, ncols)
vgg_model.summary()

# prediction matrix
vgg_act = vgg_model.predict(this_input, steps=1)

###############################################################################################
