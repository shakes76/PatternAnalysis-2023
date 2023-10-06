#%%
#import von libraries
import numpy as np
import os
import tensorflow as tf
from dataset import image_list
from dataset import load_images


#%%

#define varibles
height = 32
width = 32
dimension = 1

#define paths
path_train_images_AD = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/train/AD/"
path_train_images_NC = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/train/NC/"
path_test_images_AD = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/AD/"
path_test_images_NC = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/NC/"

#%%
#load all image names of the different train and test folders
list_train_AD, number_train_AD = image_list(path_train_images_AD)
list_train_NC, number_train_NC = image_list(path_train_images_NC)
list_test_AD, number_test_AD = image_list(path_test_images_AD)
list_test_NC, number_test_NC = image_list(path_test_images_NC)
# %%

#load all the images from the folder
#define empty list
train_images_AD = []
train_images_NC = []

#load images into list as a numpy array
train_images_NC = load_images(path_train_images_NC, list_train_NC, number_train_NC, height, width)
train_images_AD = load_images(path_train_images_AD, list_train_AD, number_train_AD, height, width)
# %%

