#%%
#import von libraries
import numpy as np
import os
import tensorflow as tf
from dataset import image_list
from dataset import load_images
import random
from sklearn.model_selection import train_test_split
from modules import siamese_network 

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

#create data with the correct shape for the neural network
train_pair_label = []
train_pair_image = []

#creates AD and NC as well as AD and AD image pairs
#AD and AD image pairs get the label 1
#AD and NC image pairs get the label 0 
for x in range(0,number_train_AD):
    random_number_AD = random.randint(0, number_train_AD-1)
    random_number_NC = random.randint(0, number_train_NC-1)
    image1 = np.expand_dims(train_images_AD[x,:,:,:], 0)
    image2 = np.expand_dims(train_images_AD[random_number_AD,:,:,:], 0)
    con_AD = np.concatenate((image1, image2), axis=0)
    train_pair_image.append(con_AD)
    train_pair_label.append(1)
    image3 = np.expand_dims(train_images_NC[random_number_NC,:,:,:], 0) #[2,32,32,1]
    con_NC = np.concatenate((image1, image3), axis=0)
    train_pair_image.append(con_NC)
    train_pair_label.append(0)

#creates NC and NC as well as NC and AD image pairs
#NC and NC image pairs get the label 1
#NC and AD image pairs get the label 0 
for x in range(0,number_train_NC):
    random_number_AD = random.randint(0, number_train_AD-1)
    random_number_NC = random.randint(0, number_train_NC-1)
    image1 = np.expand_dims(train_images_NC[x,:,:,:], 0)
    image2 = np.expand_dims(train_images_NC[random_number_NC,:,:,:], 0)
    con_NC = np.concatenate((image1, image2), axis=0)
    train_pair_image.append(con_NC)
    train_pair_label.append(1)
    image3 = np.expand_dims(train_images_AD[random_number_AD,:,:,:], 0) #[2,32,32,1]
    con_AD = np.concatenate((image1, image3), axis=0)
    train_pair_image.append(con_AD)
    train_pair_label.append(0)

train_pair_image_array = np.array(train_pair_image)
train_pair_label_array = np.array(train_pair_label)
#%%

x_train, x_valid, y_train, y_valid = train_test_split(train_pair_image_array, train_pair_label_array, test_size=0.2,random_state=42)

#%%

model = siamese_network(height,width,dimension)
model.fit(x=[x_train[:, 0, :, :], x_train[:, 1, :, :]],
          y=y_train,
          validation_data=([x_valid[:, 0, :, :],
                            x_valid[:, 1, :, :]],
                           y_valid),
          epochs=20,
          batch_size=32)
# %%
