import numpy as np

# FUNCTIONS
def get_vgg(nrows, ncols, out_layer=None):
    from tensorflow.keras.models import Model
    from tensorflow.keras import applications
    vgg_model = applications.VGG16(
                                weights="imagenet",
                                include_top=False,
                                input_shape=(nrows, ncols, 3))
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # get all outputs if not given any targets
    if not out_layer:
        outputs = [layer_dict[out].output for out in layer_dict.keys()]
        outputs = outputs[1:]
    else:
        outputs = [layer_dict[out].output for out in out_layer]

    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = False
    # model.compile(loss='mse', optimizer='adam')
    return model                                    

# CONTENT LOSS
def content_loss_np(vgg_y_pred, vgg_y_true):
    # define perceptual loss based on VGG16
    # print(vgg_y_pred.shape)
    return (l2_loss_np(vgg_y_pred, vgg_y_true))

def l2_loss_np(y_pred, y_true):
    ndim = len(y_pred.shape)
    # print(ndim)
    total = np.mean(np.square(y_pred-y_true), axis=(1, 2))
    if ndim == 4:
        total = np.sum(total, axis=-1)
    return total

def normalize_layer(vgg_tensor, eps=np.finfo(np.float32).eps):
    return vgg_tensor/np.sqrt(np.sum(np.square(vgg_tensor), axis=-1, keepdims=True)+eps)

def get_LPIPS(im1,
            im2,
            VGG_LAYERS = ['block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3',
            'block5_conv3']
            ):
    
    #
    [nrows, ncols] = im1.shape[1:3]
    
    # VGG16
    vgg_model = get_vgg(nrows, ncols, out_layer = VGG_LAYERS)
    vgg_model.summary()

    # prediction matrix
    vgg_pred = vgg_model.predict(im1, steps=1)
    vgg_input = vgg_model.predict(im2, steps=1)

    all_losses = []
    for i_layer in range(len(vgg_pred)):
        all_losses.append(content_loss_np(normalize_layer(vgg_pred[i_layer]), 
                                            normalize_layer(vgg_input[i_layer])))
    return np.asarray(all_losses)
        