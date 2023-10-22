import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.image import resize, decode_jpeg, decode_png
from tensorflow.io import read_file
import glob
from sklearn.utils import shuffle

# Define paths
DATA_PATH = 'C:/Users/keefe/Documents/COMP3710/Task1/DATA/'
TRAIN_PATH = '/ISIC2018_Task1-2_Training_Input_x2/'
MASK_PATH = '/ISIC2018_Task1_Training_GroundTruth_x2/'

# Initialise split as ratio of 70:15:15 (i.e. 1816:389:389)
train_split_size = 1816
test_val_split_size = 389


def pre_process(image, mask_image):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_image)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (256, 256))
    mask = mask == [0, 255]
    
    return img, mask


def load_data():
    # Load data from Dataset
    img_paths = sorted(glob.glob(DATA_PATH + TRAIN_PATH + '/*.jpg'))
    mask_paths = sorted(glob.glob(DATA_PATH + MASK_PATH + '/*.png'))

    # Shuffle data to randomise / prevent overfitting
    img_paths, mask_paths = shuffle(img_paths, mask_paths)

    # Split into Train, Test, Validation
    train_img = img_paths[:train_split_size]
    train_mask = mask_paths[:train_split_size]

    test_img = img_paths[train_split_size:train_split_size + test_val_split_size]
    test_mask = mask_paths[train_split_size:train_split_size + test_val_split_size]

    val_img = img_paths[train_split_size + test_val_split_size:]
    val_mask = mask_paths[train_split_size + test_val_split_size:]

    # Load into tensorflow dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_img, train_mask))
    test_data = tf.data.Dataset.from_tensor_slices((test_img, test_mask))
    val_data = tf.data.Dataset.from_tensor_slices((val_img, val_mask))

    train_data = train_data.map(pre_process)
    test_data = test_data.map(pre_process)
    val_data = val_data.map(pre_process)

    return train_data, test_data, val_data
