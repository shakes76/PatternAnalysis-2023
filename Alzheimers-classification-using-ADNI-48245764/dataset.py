import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'Alzheimers-classification-using-ADNI-48245764/AD_NC',
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


class_names = dataset.class_names
class_names


