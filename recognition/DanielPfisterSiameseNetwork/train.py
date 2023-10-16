#%%
#import von libraries
import numpy as np
import os
import tensorflow as tf
from dataset import image_list
from dataset import split_data
from dataset import load_images
from dataset import load_images_train_generator
from dataset import valid_order
from dataset import load_images_valid_generator
import random

from sklearn.model_selection import train_test_split
from modules import siamese_network 




#%%
#define varibles
height = 32
width = 32
dimension = 1
batch_size = 32

#define paths
path_train_images_AD = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/train/AD/"
path_train_images_NC = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/train/NC/"

path_test_images_AD = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/AD/"
path_test_images_NC = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/NC/"

#%%
#load all image names of the different train and test folders
list_train_AD, number_AD = image_list(path_train_images_AD)
list_train_NC, number_NC = image_list(path_train_images_NC)

list_test_AD, number_test_AD = image_list(path_test_images_AD)
list_test_NC, number_test_NC = image_list(path_test_images_NC)


# %%
split_ratio_training = 0.8
train_data_AD, valid_data_AD, number_train_AD, number_valid_AD = split_data(list_train_AD, split_ratio_training)
train_data_NC, valid_data_NC, number_train_NC, number_valid_NC = split_data(list_train_NC, split_ratio_training)
# %%
#define order of validation images
valid_data_order_AD = valid_order(valid_data_AD)
valid_data_order_NC = valid_order(valid_data_NC)

#%%

#load all the images from the folder
#define empty list
train_images_AD = []
train_images_NC = []

#load images into list as a numpy array
train_images = load_images_train_generator(path_train_images_AD, path_train_images_NC, train_data_AD, train_data_NC, number_train_AD, number_train_NC, height, width, batch_size)
valid_images = load_images_valid_generator(path_train_images_AD, path_train_images_NC, valid_data_AD, valid_data_NC, number_valid_AD, number_valid_NC, valid_data_order_AD, valid_data_order_NC, height, width, batch_size)

#%%

model = siamese_network(height,width,dimension)
history = model.fit(x=train_images,
                            validation_data = valid_images,
                            steps_per_epoch = len(train_data_AD)//batch_size,
                            validation_steps = len(valid_data_AD)//batch_size,
                            shuffle = False, epochs=20)
# %%
