import tensorflow as tf
#import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def get_train_dataset():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'recognition/VQ-VAE-46495408/keras_png_slices_data/keras_png_slices_train',
        label_mode=None,
        color_mode='grayscale',
        batch_size=128,
        image_size=(256, 256)
    )
    normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-0.5)
    train_ds = train_ds.map(lambda x: normalization_layer(x))
    return train_ds

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

def preview_images():
    train_ds = get_train_dataset()
    plt.figure(figsize=(10,10))
    plt.title("Sample of images from OASIS train dataset")
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(next(iter(train_ds))[i])
        plt.axis('off')
    plt.show()
    
def get_training_variance():
    # Calculate the mean value
    train_ds = get_train_dataset()
    train_sum = 0
    num_samples = 0
    for batch in train_ds:
        num_samples += len(batch)
        train_sum += tf.reduce_sum(batch)
    train_mu = train_sum / (num_samples * 256 ** 2)
    # Calculate the variance
    variance = 0
    for batch in train_ds:
        variance += tf.reduce_sum((batch - train_mu) ** 2)
    variance /= (num_samples * 256 ** 2) - 1
    return variance
    
print(get_training_variance())