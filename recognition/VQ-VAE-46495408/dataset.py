import tensorflow as tf
import matplotlib.pyplot as plt

def get_train_dataset():
    """
    Load the train data from the data folder
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_train',
        label_mode=None,
        color_mode='grayscale',
        batch_size=128,
        image_size=(256, 256)
    )
    # Normalize the dataset
    normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-0.5)
    train_ds = train_ds.map(lambda x: normalization_layer(x))
    return train_ds

def get_validate_dataset():
    """
    Load the train data from the data folder
    """
    validate_ds = tf.keras.utils.image_dataset_from_directory(
        'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_validate',
        label_mode=None,
        color_mode='grayscale',
        #batch_size=128,
        image_size=(256, 256)
    )
    # Normalize the dataset
    normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-0.5)
    validate_ds = validate_ds.map(lambda x: normalization_layer(x))
    return validate_ds

def get_test_dataset():
    """
    Load the test data from the data folder
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_test',
        label_mode=None,
        color_mode='grayscale',
        batch_size=128,
        image_size=(256, 256)
    )
    # Normalize the dataset
    normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-0.5)
    test_ds = test_ds.map(lambda x: normalization_layer(x))
    return test_ds

def preview_images():
    """
    Plot images in the dataset
    """
    train_ds = next(iter(get_train_dataset().take(1)))
    plt.figure(figsize=(10,10))
    plt.title("Sample of images froxm OASIS train dataset")
    
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(train_ds[i])
        plt.axis('off')
    plt.show()
    
def get_dataset_variance(dataset):
    """
    Calculate the variance of the dataset
    """
    # Calculate the mean value
    sum = 0
    num_samples = 0
    for batch in dataset:
        num_samples += len(batch)
        sum += tf.reduce_sum(batch)
    mu = sum / (num_samples * 256 ** 2)
    
    # Calculate the variance
    variance = 0
    for batch in dataset:
        variance += tf.reduce_sum((batch - mu) ** 2)
    variance /= (num_samples * 256 ** 2) - 1
    return variance