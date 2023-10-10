import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing


# Constant Parameters
seed = 420
train_path = '/home/groups/comp3710/OASIS/keras_png_slices_seg_train/'
test_path = '/home/groups/comp3710/OASIS/keras_png_slices_seg_test/'
validation_path = '/home/groups/comp3710/OASIS/keras_png_slices_seg_validate/'


def normalise_dataset(dataset):
    # Normalize the pixel values to be in the range [0, 1]
    rescale = preprocessing.Rescaling(1.0 / 255.0)
    normalised_dataset = dataset.map(lambda x, y: (rescale(x), y))
    return normalised_dataset

def load_test_data(batch_size, image_size):
    test_ds = keras.utils.image_dataset_from_directory(
        directory=test_path,
        labels=None,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        color_mode="grayscale",
    )

    return normalise_dataset(test_ds)

def load_train_data(batch_size, image_size):
    train_ds = keras.utils.image_dataset_from_directory(
        directory=train_path,
        labels=None,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        color_mode="grayscale",
    )

    return normalise_dataset(train_ds)

def load_validation_data(batch_size, image_size):
    validation_ds = keras.utils.image_dataset_from_directory(
        directory=validation_path,
        labels=None,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        color_mode="grayscale",
    )

    return normalise_dataset(validation_ds)
