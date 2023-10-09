import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
Function used to load ADNI datasets. 
Assumes image size is 256x240

Outputs:
    train_generator - the training set converted to a generator, split into AD and NC classes
    labels - the test set converted to a generator, split into AD and NC classes
'''
def load_dataset():
    # Define the training ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,  # Normalize pixel values
                                rotation_range=20,   # Ratates each image by up to 20 degrees
                                width_shift_range=0.2, # Changes the width of each image
                                height_shift_range=0.2, # Changes the height of each image
                                horizontal_flip=True, # Randomly flips each image horizontally
                                vertical_flip=True) # Randomly flips each image vertically

    # Create training generator
    train_generator = train_datagen.flow_from_directory("AD_NC/train", # Path for train set
                                                target_size=(256, 240),  # Set the desired image size
                                                batch_size=32, # Det the batch size
                                                class_mode='categorical')  # Split into AD and NC classes
    
    # Define the testing ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1.0/255.0) # Normalize pxel values

    # Create testing generator
    test_generator = test_datagen.flow_from_directory("AD_NC/test", # Path for test set
                                                    target_size=(256, 240), # Det the desired image size
                                                    batch_size=32, # Set the batch size
                                                    class_mode='categorical') # Split into AD and NC classes
    return train_generator, test_generator



"""
Main method used to see information about the data generators
"""
def main():
    train_generator, test_generator = load_dataset()

    # Print information about the training generator
    print("Training Generator:")
    print("Number of batches in training generator:", len(train_generator))
    print("Number of samples in the training dataset:", train_generator.samples)
    print("Number of classes:", train_generator.num_classes)
    print("Class labels and their corresponding indices:", train_generator.class_indices)
    print("Batch size:", train_generator.batch_size)
    print("Image shape:", train_generator.image_shape)

    # Print information about the test generator
    print("\nTest Generator:")
    print("Number of batches in test generator:", len(test_generator))
    print("Number of samples in the test dataset:", test_generator.samples)
    print("Number of classes:", test_generator.num_classes)
    print("Class labels and their corresponding indices:", test_generator.class_indices)
    print("Batch size:", test_generator.batch_size)
    print("Image shape:", test_generator.image_shape)



if __name__=='__main__':
    main()