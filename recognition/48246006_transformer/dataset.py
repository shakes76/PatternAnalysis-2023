import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(img_height, img_width, batch_size):
    # Define the paths to the training and test data directories
    train_data_dir = '/Users/saakshigupta/Desktop/code/AD_NC/train'  
    test_data_dir = '/Users/saakshigupta/Desktop/code/AD_NC/test'    
    num_classes = 2  # Number of classes in your dataset (e.g., 2 for binary classification)

    # Data preprocessing for the training set
     train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Rescale pixel values to [0, 1]
        rotation_range=20,  # Rotate images up to 20 degrees
        width_shift_range=0.2,  # Shift image width by up to 20%
        height_shift_range=0.2,  # Shift image height by up to 20%
        horizontal_flip=True,  # Flip images horizontally
        fill_mode='nearest'  # Fill mode for new pixels
    )

    # Create a generator for the training data
    train_data_gen = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  # 'binary' or 'categorical'
    )


    # Data preprocessing for the test set
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale pixel values for the test set

    # Create a generator for the test data
     test_data_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',  
    )

    # Return the data generators for training and testing
    return train_data_gen, test_data_gen
