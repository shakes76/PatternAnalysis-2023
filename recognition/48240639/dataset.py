"""
Created on Wednesday October 18 
ADNI Dataset and Data Loaders

This code defines a custom dataset class, ADNIDataset, for loading and processing
ADNI dataset images for use in Siamese Network training and testing. It also provides
functions to get train and test datasets from a specified data path.

@author: Aniket Gupta 
@ID: s4824063

"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ADNIDataset(keras.utils.Sequence):
    def __init__(self, data_path, batch_size=32, shuffle=True, mode='train'):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode

        self.transform = ImageDataGenerator(rescale=1.0/255.0)

        self.ad_path = os.path.join(data_path, 'AD')
        self.nc_path = os.path.join(data_path, 'NC')

        self.ad_images = [os.path.join(self.ad_path, img) for img in os.listdir(self.ad_path)]
        self.nc_images = [os.path.join(self.nc_path, img) for img in os.listdir(self.nc_path)]

        self.indexes = list(range(len(self.ad_images) if mode == 'AD' else len(self.nc_images)))

        if shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.mode == 'AD':
            batch_images = [tf.keras.preprocessing.image.load_img(self.ad_images[i], target_size=(224, 224)) for i in batch_indexes]
            labels = [1] * len(batch_images)
        else:
            batch_images = [tf.keras.preprocessing.image.load_img(self.nc_images[i], target_size=(224, 224)) for i in batch_indexes]
            labels = [0] * len(batch_images)

        batch_images = [self.transform.img_to_array(img) for img in batch_images]
        batch_images = tf.convert_to_tensor(batch_images)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        return batch_images, labels

def get_train_dataset(data_path, batch_size=32):
    train_dataset = ADNIDataset(os.path.join(data_path, 'train'), batch_size=batch_size, mode='train')
    return train_dataset

def get_test_dataset(data_path, batch_size=32):
    test_dataset = ADNIDataset(os.path.join(data_path, 'test'), batch_size=batch_size, mode='test')
    return test_dataset
