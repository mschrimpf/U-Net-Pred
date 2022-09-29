from pathlib import Path

import numpy as np

from model.model_fn import build_compile_model
from model_tools.activations.keras import KerasWrapper, load_images
from model_tools.check_submission import check_models

"""
Template module for a base model submission to brain-score
"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['u-pred-net']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'u-pred-net'
    params = {
        # https://github.com/uranc/U-Net-Pred/blob/e9170a8edca7ab3d5baf0b632c614b9bc8a27d1a/sample_nat_im_demo.ipynb
        'weight_dir': str(Path(__file__).parent / '..' / 'weights.last.h5'), 'lr': 1.56 * 1e-4,
        'p_content': 0, 'p_style': 0, 'p_hole': 0, 'p_tv': 0,
        'p_fft_s': 0.002, 'p_fft_c': 8.79,  # 'p_fft_s': 0.330, 'p_fft_c': 505.47,
        'p_fft_abs': 1, 'p_fft_log': 0.026, 'p_fft_phase': 6.45e-05,  # .0556
        'b_size': 4, 'current_epoch': 0,
        'n_epoch': 150000,
        'exp_dir': 'experiments/',
        'inst_norm': False,
        'input_size': 224
    }
    model = build_compile_model(params)
    basemodel = KerasWrapper(model=model, preprocessing=preprocessing, batch_size=1)
    return basemodel


def preprocessing(stimulus_paths):
    images = load_images(stimulus_paths, image_size=224)
    # from https://github.com/uranc/U-Net-Pred/blob/e9170a8edca7ab3d5baf0b632c614b9bc8a27d1a/sample_nat_im_demo.ipynb
    normalizer = [103.939, 116.779, 123.68]
    images = [image[:, :] - normalizer for image in images]
    images = [np.expand_dims(image, axis=0) for image in images]

    im_mask = [np.zeros(image.shape) for image in images]
    model_input = [images] + [im_mask]

    return model_input


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    assert name == 'u-pred-net'
    return ['re_lu'] + [f're_lu_{num}' for num in range(1, 20)]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    assert model_identifier == 'u-pred-net'
    return """@article{uran2021predictive,
            title={Predictive coding of natural images by V1 activity revealed by self-supervised deep neural networks},
            author={Uran, Cem and Peter, Alina and Lazar, Andreea and Barnes, William and Klon-Lipok, Johanna and Shapcott, Katharine A and Roese, Rasmus and Fries, Pascal and Singer, Wolf and Vinck, Martin},
            journal={bioRxiv},
            pages={2020--08},
            year={2021},
            publisher={Cold Spring Harbor Laboratory}
            }
            """


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
