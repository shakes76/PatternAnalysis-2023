from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, reshape
from PIL import Image

NMALISE = 255.0
CENTRE = 0.5
IMAGE_DIM = 80
ENC_IN_SHAPE = (80, 80, 1)
GREYSCALE = "gray"  # Use "gray" for grayscale images

SPLITS = 3
BASE = "keras_png_slices_data/keras_png_slices_"
TRAINING = BASE + "train/*"
TESTING = BASE + "test/*"
VALIDATION = BASE + "validate/*"

def get_ttv():

    """
    Read in the training/testing/validation datasets from local files.
    Mostly repurposed from demo 2

    return	- the training, testing, and validation datasets
    """

    srcs = (glob(TRAINING), glob(TESTING), glob(VALIDATION))
    dsets = ([[]] * SPLITS)

    for i, src in enumerate(srcs):
        for path in src:
            base = Image.open(path)
            scale = base.resize((IMAGE_DIM, IMAGE_DIM))
            npify = reshape(scale, ENC_IN_SHAPE)
            dsets[i].append(npify)

    as_arrs = tuple([array(d) for d in dsets])
    train_dset, test_dset, val_dset = as_arrs

    return (train_dset, test_dset, val_dset)

def normalise(dset):
    """Scale and center a dataset to have all values within [-0.5, 0.5]"""
    return dset / NMALISE - CENTRE

def preview(dataset, n):

    """
    Show the first n^2 images of the dataset in an n x n grid

    dataset	- training / testing / validation dataset to preview
    n		- length of the preview square grid
    """

    for i in range(n):
        for j in range(n):
            ind = (n * i) + j + 1
            if ind <= len(dataset):
                plt.subplot(n, n, ind)
                plt.axis("off")
                plt.imshow(dataset[ind - 1], cmap=GREYSCALE)

    plt.show()

if __name__ == "__main__":

    # Load the training/testing/validation datasets
    train_dset, test_dset, val_dset = get_ttv()

    # Normalize the datasets
    train_dset = normalise(train_dset)
    test_dset = normalise(test_dset)
    val_dset = normalise(val_dset)

    # Preview the training dataset
    preview(train_dset, 4)
