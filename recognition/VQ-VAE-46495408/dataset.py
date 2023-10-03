import tensorflow as tf

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

print(test_ds)