import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import read_image, read_mask

# preset img size
H = 256
W = 256

def load_data(dataset_path, split):
    """
    Loads the data and splits it into training, validation, and testing sets.

    Parameters:
    dataset_path (str): The path to the dataset.
    split (float): The % of the data to be used for validation and testing and training.

    Returns:
    Tuple: Training, validation, and testing sets
    """
    # Load the images and masked based on their extension
    images = sorted(glob(os.path.join(dataset_path, "ISIC-2017_Training_Data", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    test_size = int(len(images) * split)
    # Create first split based on test size
    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)
    # Second split based on test size on the train datasets
    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def tf_parse(x, y):
    """
    Parses the image and mask using TensorFlow.

    Parameters:
    x (str): The image path.
    y (str): The mask path.

    Returns:
    Tuple: The image and mask as TensorFlow constants.
    """
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    # wrap function to tf 
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    # Set shapes based on RGB or Grayscale
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size):
    """
    Creates a TensorFlow dataset for the provided data.

    Parameters:
    X (list): List of image paths.
    Y (list): List of mask paths.
    batch_size (int): The batch size for the dataset.

    Returns:
    Dataset: The TensorFlow dataset with the images and masks.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    # Procces batches on CPU while GPU in use (consumer/prod overlap)
    dataset = dataset.prefetch(10)
    return dataset

def get_steps(data, batch_size):
    """
    Calculates the number of steps between a dataset and batch size.

    Parmeters:
    data (list): The dataset.
    batch_size (int): The batch size.

    Returns:
    int: Steps needed for the dataset with the given batch size.
    """
    steps = len(data)//batch_size

    if len(data) % batch_size != 0:
        steps += 1

    return steps