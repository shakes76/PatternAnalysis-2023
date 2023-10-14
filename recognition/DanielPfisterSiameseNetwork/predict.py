# %%

import tensorflow as tf
from dataset import image_list
from dataset import load_images
import random
import numpy as np
import tensorflow.keras.backend as K


#define varibles
height = 128
width = 128
dimension = 1

#load the trained weights of the neural network
model = tf.keras.saving.load_model("C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/PatternAnalysis-2023/recognition/DanielPfisterSiameseNetwork/model1.h5")

# %%
path_test_images_AD = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/AD/"
path_test_images_NC = "C:/Users/Daniel/Desktop/Studium/UQ/5.Semester/COMP3710/Assignment/LabReport/Dataset/ADNI/AD_NC/test/NC/"

list_test_AD, number_test_AD = image_list(path_test_images_AD)
list_test_NC, number_test_NC = image_list(path_test_images_NC)


test_images_AD = []
test_images_NC = []

#load images into list as a numpy array
test_images_AD = load_images(path_test_images_AD, list_test_AD, number_test_AD, height, width)
test_images_NC = load_images(path_test_images_NC, list_test_NC, number_test_NC, height, width)


# %%

#create data with the correct shape for the neural network
test_pair_label = []
test_pair_image = []

#creates AD and NC as well as AD and AD image pairs
#AD and AD image pairs get the label 1
#AD and NC image pairs get the label 0 
for x in range(0,number_test_AD):
    random_number_AD = random.randint(0, number_test_AD-1)
    random_number_NC = random.randint(0, number_test_NC-1)
    image1 = np.expand_dims(test_images_AD[x,:,:,:], 0)
    image2 = np.expand_dims(test_images_AD[random_number_AD,:,:,:], 0)
    con_AD = np.concatenate((image1, image2), axis=0)
    test_pair_image.append(con_AD)
    test_pair_label.append(1)
    image3 = np.expand_dims(test_images_NC[random_number_NC,:,:,:], 0) #[2,32,32,1]
    con_NC = np.concatenate((image1, image3), axis=0)
    test_pair_image.append(con_NC)
    test_pair_label.append(0)

#creates NC and NC as well as NC and AD image pairs
#NC and NC image pairs get the label 1
#NC and AD image pairs get the label 0 
for x in range(0,number_test_NC):
    random_number_AD = random.randint(0, number_test_AD-1)
    random_number_NC = random.randint(0, number_test_NC-1)
    image1 = np.expand_dims(test_images_NC[x,:,:,:], 0)
    image2 = np.expand_dims(test_images_NC[random_number_NC,:,:,:], 0)
    con_NC = np.concatenate((image1, image2), axis=0)
    test_pair_image.append(con_NC)
    test_pair_label.append(1)
    image3 = np.expand_dims(test_images_AD[random_number_AD,:,:,:], 0) #[2,32,32,1]
    con_AD = np.concatenate((image1, image3), axis=0)
    test_pair_image.append(con_AD)
    test_pair_label.append(0)

test_pair_image_array = np.array(test_pair_image)
test_pair_label_array = np.array(test_pair_label)


# %%

# print the test accuracy
metrics = model.evaluate([test_pair_image_array[:, 0, :, :], test_pair_image_array[:, 1, :, :]], test_pair_label_array)
print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))
# %%
