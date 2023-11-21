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
from modules import siamese_network 
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

#%%
#define varibles
height = 128
width = 128
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
#split data into training and validation dataset
split_ratio_training = 0.8
train_data_AD, valid_data_AD, number_train_AD, number_valid_AD = split_data(list_train_AD, split_ratio_training)
train_data_NC, valid_data_NC, number_train_NC, number_valid_NC = split_data(list_train_NC, split_ratio_training)
# %%
#define order of validation images
valid_data_order_AD = valid_order(valid_data_AD)
valid_data_order_NC = valid_order(valid_data_NC)

#%%
#load all the images from the folder

#define list
train_images_AD = []
train_images_NC = []

#load images into list as a numpy array
train_images = load_images_train_generator(path_train_images_AD, path_train_images_NC, train_data_AD, train_data_NC, number_train_AD, number_train_NC, height, width, batch_size)
valid_images = load_images_valid_generator(path_train_images_AD, path_train_images_NC, valid_data_AD, valid_data_NC, number_valid_AD, number_valid_NC, valid_data_order_AD, valid_data_order_NC, height, width, batch_size)

#%%
#define callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=6,verbose=0,mode="auto",baseline=None,restore_best_weights=True,start_from_epoch=5)
#define model
siamese_model = siamese_network(height,width,dimension)
#training of the neural network
history = siamese_model.fit(x=train_images,
                            validation_data = valid_images,
                            steps_per_epoch = len(train_data_AD)//batch_size,
                            validation_steps = len(valid_data_AD)//batch_size,
                            shuffle = False, epochs=50, callbacks=[callback])


# %%
# Plot training and validation accuracy per epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Get number of epochs
epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Plot training and validation loss per epoch
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# %%
#saves the model
siamese_model.save('C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/PatternAnalysis-2023/recognition/DanielPfisterSiameseNetwork/model1.h5')

#%%
#load the trained weights of the neural network
siamese_model = tf.keras.saving.load_model("C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/PatternAnalysis-2023/recognition/DanielPfisterSiameseNetwork/model1.h5")

# %%

# validate the model with the validate dataset
metrics_valid = siamese_model.evaluate(valid_images,steps = len(train_data_AD)//batch_size)
print('Loss of {} and Accuracy is {} %'.format(metrics_valid[0], metrics_valid[1] * 100))


# %%
# Test model with the test dataset
#define the order of the test images
test_data_order_AD = valid_order(list_test_AD)
test_data_order_NC = valid_order(list_test_NC)

#load test images
test_images = load_images_valid_generator(path_test_images_AD, path_test_images_NC, list_test_AD, list_test_NC, number_test_AD, number_test_NC,test_data_order_AD, test_data_order_NC,height, width, batch_size= batch_size)

#test the model with test images
metrics_test = siamese_model.evaluate(test_images,steps = len(list_test_AD)//batch_size)
print('Loss of {} and Accuracy is {} %'.format(metrics_test[0], metrics_test[1] * 100))


# %%
