#%%
#import von libraries
import os
import numpy as np
import tensorflow as tf


#%%
#define the function to read all the image names from folder 
#und count the image number
def image_list(path_folder_images):
    list_image = sorted(os.listdir(path_folder_images))
    image_count = len(list_image)
    print("Number of images: " + str(image_count))
    
    return list_image, image_count

#%% 
# splits the dataset in training and validation 
def split_data(list_image, split_ratio):
    train_number_split = int(len(list_image)*split_ratio)
    print(train_number_split)
    train_data = list_image[:train_number_split] #takes the first to train_number and defines them as training images
    valid_data = list_image[train_number_split:] #takes the train_number to the last element and defines them as validation images
    print(train_data)
    print(valid_data)
    number_train = len(train_data)
    number_valid = len(valid_data)
    print("Number of training images: " + str(number_train))
    print("Number of validation images AD: " + str(number_valid))
    return train_data, valid_data, number_train, number_valid



#%%
# define the function to load the images
def load_images(path_folder_images, list_images, number_images, height, width):
    image_array = []
    a = 0
    for a in range(0, number_images):
        path_image = path_folder_images + list_images[a] # create the path of the image
        img = tf.keras.utils.load_img(path_image, target_size=(height, width), color_mode="grayscale") # load the image
        img = tf.keras.utils.img_to_array(img) # convert the image into an array
        img = img / 255 #normalize the image
        image_array.append(img) # append the array image to the list

    image_array = np.array(image_array) # create a numpy array
    return image_array
# %%


