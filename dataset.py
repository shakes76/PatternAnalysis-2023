import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)

H = 256
W = 256

def create_dir(path):
    """
    Create a directory if it does not exist.

    Parameters:
    - path (str): The path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    """
    Shuffle the input arrays x and y.

    Parameters:
    - x: Input array 1.
    - y: Input array 2.

    Returns:
    - Tuple: Shuffled arrays x and y.
    """
    x, y = shuffle(x, y, random_state=1)
    return x, y

def load_data(split=0.15):
    """
    Load and split image and mask data into training, validation, and test sets.

    Parameters:
    - split (float): The proportion of data to be used for validation and test sets.

    Returns:
    - Tuple: Three tuples containing (train_x, train_y), (valid_x, valid_y), and (test_x, test_y).
    """
    images = sorted(glob(os.path.join(r"ISIC2018_Task1-2_Training_Input_x2", "*.jpg")))
    masks = sorted(glob(os.path.join(r"ISIC2018_Task1_Training_GroundTruth_x2", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=1)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=1)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=1)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=1)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    """
    Read and preprocess an image from the specified path.

    Parameters:
    - path (str): The path of the image file.

    Returns:
    - np.ndarray: The preprocessed image.
    """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x  # (256, 256, 3)

def read_mask(path):
    """
    Read and preprocess a mask image from the specified path.

    Parameters:
    - path (str): The path of the mask image file.

    Returns:
    - np.ndarray: The preprocessed mask image.
    """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x  # (256, 256, 1)

def data_augmentation(image, mask):
    """
    Apply data augmentation to an input image and mask.

    Parameters:
    - image (tf.Tensor): Input image.
    - mask (tf.Tensor): Input mask.

    Returns:
    - Tuple: Augmented image and mask.
    """
    # Add your data augmentation here using tf.image functions
    # Example: Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Add more augmentation techniques as needed

    return image, mask

def tf_parse(x, y, augment=True):
    """
    Parse and preprocess input data for TensorFlow dataset.

    Parameters:
    - x: Input image path.
    - y: Input mask path.
    - augment (bool): Flag indicating whether to apply data augmentation.

    Returns:
    - Tuple: Parsed and preprocessed image and mask.
    """
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)

        if augment:
            # Apply data augmentation
            x, y = data_augmentation(x, y)

        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch, augment=True):
    """
    Create a TensorFlow dataset from input data.

    Parameters:
    - X: Input images.
    - Y: Input masks.
    - batch (int): Batch size.
    - augment (bool): Flag indicating whether to apply data augmentation.

    Returns:
    - tf.data.Dataset: TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf_parse(x, y, augment), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
