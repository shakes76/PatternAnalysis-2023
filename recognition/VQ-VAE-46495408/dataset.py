import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_ds = tf.keras.utils.image_dataset_from_directory(
    'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_train',
    label_mode=None,
    color_mode='grayscale',
    batch_size=128,
    image_size=(256, 256)
)

validate_ds = tf.keras.utils.image_dataset_from_directory(
    'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_validate',
    label_mode=None,
    color_mode='grayscale',
    batch_size=128,
    image_size=(256, 256)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_test',
    label_mode=None,
    color_mode='grayscale',
    batch_size=128,
    image_size=(256, 256)
)

normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-0.5)
train_ds = train_ds.map(lambda x: normalization_layer(x))
plt.figure(figsize=(10,10))
plt.title("Sample of images from OASIS train dataset")
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(next(iter(train_ds))[i])
    plt.axis('off')
plt.show()