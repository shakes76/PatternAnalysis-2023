# imports
from tensorflow import keras

# get training data
trainDataDir = './ADNI_AD_NC_2D/AD_NC/train'
testDataDir = './ADNI_AD_NC_2D/AD_NC/test'
classes = ['AD', 'NC']
imageSize = 256


# load training and validation data from directory
def create_train_data(img_size, batch_size):
    trainDataset = keras.utils.image_dataset_from_directory(
        directory=trainDataDir,  # directory location
        labels='inferred',  # label given by directory name
        label_mode='binary',  # label is either AD or NC (binary labeling)
        batch_size=batch_size,  # set batch size to
        image_size=(img_size, img_size),  # image size after resizing
        subset='validation',  # make validation subset of data
        validation_split=0.25  # make 1/4 of data into validation subset
    )
    return trainDataset


# load testing data
def create_test_data(img_size, batch_size):
    testDataset = keras.utils.image_dataset_from_directory(
        directory=testDataDir,  # directory location
        labels='inferred',  # label given by directory name
        label_mode='binary',  # label is either AD or NC (binary labeling)
        batch_size=batch_size,  # set batch size to
        image_size=(img_size, img_size),  # image size after resizing
    )
    return testDataset
