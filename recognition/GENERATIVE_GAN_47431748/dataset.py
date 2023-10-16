"""
Data loader + preprocessing
Jack Cashman - 47431748
"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

DATA_PATH = r'C:\Users\jackc\OneDrive\Desktop\UQ\UQ23S2\COMP3710_PROJ\PatternAnalysis-2023\AD_NC'
IMG_DIM = 256
BATCH_SIZE = 32
SHIFT = 0.5

def load_preprocess_image_data(path, img_dim, batch_size, shift):
    """
    Load and preprocess the image data
    :param path: Path to unzipped data
    :param img_dim: Dimension of the square images
    :param batch_size: Size of each batch
    :param shift: Normalisation const.
    :return: tf.data.Dataset object
    """
    img_data = image_dataset_from_directory(path, label_mode=None, image_size=(img_dim, img_dim), color_mode='rgb',
                                            batch_size=batch_size, shuffle=True)
    return img_data.map(lambda x: (x / float(img_dim)) - shift)