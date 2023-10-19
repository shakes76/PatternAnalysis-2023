import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(img_height, img_width, batch_size):
    train_data_dir = '/Users/saakshigupta/Desktop/code/AD_NC/train'  # Update this path
    test_data_dir = '/Users/saakshigupta/Desktop/code/AD_NC/test'    # Update this path

    num_classes = 2  # Number of classes

    # Data preprocessing for training set
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Change to 'categorical'
    )

    # Data preprocessing for the test set
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    test_data_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # Change to 'categorical'
    )

    return train_data_gen, test_data_gen
