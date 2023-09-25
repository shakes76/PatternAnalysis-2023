import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from enum import Enum
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

DATA_PATH = '/Users/samson/Documents/UQ/COMP3710/data/keras_png_slices_data/'                     # root of data dir
TRAIN_INPUT_PATH = DATA_PATH + 'keras_png_slices_train/'         # train input
VALID_INPUT_PATH = DATA_PATH + 'keras_png_slices_validate/'      # valid input
TEST_INPUT_PATH = DATA_PATH + 'keras_png_slices_test/'           # test input
VALID_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_validate/' # train target
TRAIN_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_train/'    # valid target
TEST_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_test/'      # test target
MODEL_PATH = './vqvae2.pth'         # trained model
TRAIN_TXT = './oasis_train.txt'     # info of img for train
VALID_TXT = './oasis_valid.txt'     # info of img for valid
TEST_TXT = './oasis_test.txt'       # info of img for test
GENERATED_IMG_PATH = 'gened_imgs/'

