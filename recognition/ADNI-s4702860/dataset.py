import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



'''
Function used to load ADNI datasets. 
Assumes image size is 256x240

Inputs:
    base_path - the base path (test, train) used for classifying either the train or test data

Outputs:
    data - the image converted to a tensorflow tensor
    labels - the classification label, 1 for having alzeimers, else 0
'''
def load_dataset():
    # Define the image data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0,  # Normalize pixel values
                                rotation_range=20,   # Augmentation options
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                validation_split=0.2)  # Split data into training and validation sets

    # Create separate data generators for training and validation sets
    train_generator = datagen.flow_from_directory("AD_NC/train",
                                                target_size=(256, 240),  # Set the desired image size
                                                batch_size=32,
                                                class_mode='categorical',
                                                subset='training')  # Specify 'training' or 'validation'

    test_generator = datagen.flow_from_directory("AD_NC/test",
                                                    target_size=(256, 240),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='validation')
    return train_generator, test_generator




def main():
    train_generator, test_generator = load_dataset()

    print(train_generator.shape)
    print(test_generator.shape)


    



if __name__=="__main__":
    main()