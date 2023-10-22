import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    
    training_images = 'AD_NC/train'
    
    testing_images = 'AD_NC/test'

    AD_train_files = os.listdir(os.path.join(training_images, 'AD'))
    AD_test_files = os.listdir(os.path.join(testing_images, 'AD'))
    NC_train_files = os.listdir(os.path.join(training_images, 'NC'))
    NC_test_files = os.listdir(os.path.join(testing_images, 'NC'))


    training_data = [] 

    for filename in AD_train_files:
        image = load_img(os.path.join(training_images, 'AD', filename), target_size=(140, 140, 3))
        image = img_to_array(image)
        training_data.append([image, 1])  

    for filename in NC_train_files:
        image = load_img(os.path.join(training_images, 'NC', filename), target_size=(140, 140, 3))
        image = img_to_array(image)
        training_data.append([image, 0])  


    training_images = []
    training_labels = []
    
    
    for _ in training_data:
        training_images.append(_[0])  
        training_labels.append(_[1])  
        
        



train_size = 0.8
len(dataset)*train_size

train_ds = dataset.take(54)
len(train_ds)

test_ds = dataset.skip(54)
len(test_ds)

val_size=0.1
len(dataset)*val_size

val_ds = test_ds.take(6)
len(val_ds)

test_ds = test_ds.skip(6)
len(test_ds)


# def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
#     assert (train_split + test_split + val_split) == 1
    
#     ds_size = len(ds)
    
#     if shuffle:
#         ds = ds.shuffle(shuffle_size, seed=12)
    
#     train_size = int(train_split * ds_size)
#     val_size = int(val_split * ds_size)
    
#     train_ds = ds.take(train_size)    
#     val_ds = ds.skip(train_size).take(val_size)
#     test_ds = ds.skip(train_size).skip(val_size)
    
#     return train_ds, val_ds, test_ds


# train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# len(train_ds)


# len(val_ds)


# len(test_ds)


# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
# test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
