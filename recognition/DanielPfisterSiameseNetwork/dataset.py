#%%
#import von libraries
import os
import numpy as np
import tensorflow as tf
import random


#%%
#define the function to read all the image names from folder 
#and count the image number
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
def load_images(path_folder_images, chunk_images, height, width):
    image_array = []
    for a in chunk_images:
        path_image = path_folder_images + a # create the path of the image
        img = tf.keras.utils.load_img(path_image, target_size=(height, width), color_mode="grayscale") # load the image
        img = tf.keras.utils.img_to_array(img) # convert the image into an array
        img = img / 255 #normalize the image
        image_array.append(img) # append the array image to the list

    image_array = np.array(image_array) # create a numpy array
    return image_array
# %%
#data generator training images
def load_images_train_generator(path_folder_images1, path_folder_images2,  list_images1, list_images2, number_images1, number_images2, height, width, batch_size):
    while True:
        #list_images
        random.shuffle(list_images1) #shuffels the images from the first list
        random.shuffle(list_images2) #shuffels the images from the second list
        for i in range(0, len(list_images1), batch_size): #takes the frist images and loads them into the chunk variable
            chunk_AD = list_images1[i:i + batch_size]
            chunk_NC = list_images2[i:i + batch_size]
            image_array_AD = []
            image_array_NC = []

            for j in chunk_AD:
                path_image_AD = path_folder_images1 + j # create the path of the image
                imgAD = tf.keras.utils.load_img(path_image_AD, target_size=(height, width), color_mode="grayscale") # load the images
                imgAD = tf.keras.utils.img_to_array(imgAD) # convert the image into an array
                imgAD = imgAD / 255.0 #normalize the image
                image_array_AD.append(imgAD)
            image_array_AD = np.array(image_array_AD) # create a numpy array
            
            for k in chunk_NC:
                path_image_NC = path_folder_images2 + k # create the path of the image
                imgNC = tf.keras.utils.load_img(path_image_NC, target_size=(height, width), color_mode="grayscale") # load the image
                imgNC = tf.keras.utils.img_to_array(imgNC) # convert the image into an array
                imgNC = imgNC / 255.0 #normalize the image
                image_array_NC.append(imgNC)
            image_array_NC = np.array(image_array_NC) # create a numpy array
            image1_list = []
            image2_list = []
            label_list= []

            for ad1_data, nc1_data in zip(image_array_AD, image_array_NC):
                ad2_index = random.randint(0, len(list_images1)-1)
                ad2 = list_images1[ad2_index]
                ad2 = path_folder_images1 + ad2 # create the path of the image
                ad2_data = tf.keras.utils.load_img(ad2, target_size=(height, width), color_mode="grayscale") # load the image
                ad2_data = tf.keras.utils.img_to_array(ad2_data)
                ad2_data = ad2_data / 255.0 #normalize the image

                nc2_index = random.randint(0, len(list_images2) -1)
                # print(len(list_images2), nc2_index)
                nc2 = list_images2[nc2_index]
                nc2 = path_folder_images2 + nc2 # create the path of the image
                nc2_data = tf.keras.utils.load_img(nc2, target_size=(height, width), color_mode="grayscale") # load the image
                nc2_data = tf.keras.utils.img_to_array(nc2_data)
                nc2_data = nc2_data / 255.0 #normalize the image
                #creates four pairs of images
                #AD and AD with label 1, AD and NC with label 0, 
                #NC and NC with label 1, NC and AD with label 0
                image1_list.append(ad1_data)
                image2_list.append(ad2_data)
                label_list.append(1)
                image1_list.append(ad1_data)
                image2_list.append(nc2_data)
                label_list.append(0)
                image1_list.append(nc1_data)
                image2_list.append(nc2_data)
                label_list.append(1)
                image1_list.append(nc1_data)
                image2_list.append(ad2_data)
                label_list.append(0)

            # shuffel the image pairs with the corresponding label
            zipped_list = list(zip(image1_list, image2_list, label_list))
            random.shuffle(zipped_list)
            image1_list, image2_list, label_list = zip(*(zipped_list))

            # convert to np array
            image1_array = np.array(image1_list)
            image2_array = np.array(image2_list)
            label_array = np.array(label_list)
            yield [image1_array, image2_array], label_array

#%%
# create a list of number and shuffle the list
# the shuffled list is the order of the validation images
def valid_order(valid_data):
    data_order = []
    data_order = list(range(0,len(valid_data)))
    print(data_order)
    random.shuffle(data_order)
    print(data_order)
    return data_order
# %%
#data generator validation images
def load_images_valid_generator(path_folder_images1, path_folder_images2,  list_images1, list_images2, number_images1, number_images2, shuffel_list1, shuffel_list2, height, width, batch_size):
    while True:
        for i in range(0, len(list_images1), batch_size): #takes the frist images and loads them into the chunk variable
            chunk_AD = list_images1[i:i + batch_size]
            chunk_NC = list_images2[i:i + batch_size]
            indexAD = shuffel_list1[i:i + batch_size]
            indexNC = shuffel_list2[i:i + batch_size]
            image_array_AD = []
            image_array_NC = []
            
            for j in chunk_AD:
                path_image_AD = path_folder_images1 + j # create the path of the image
                imgAD = tf.keras.utils.load_img(path_image_AD, target_size=(height, width), color_mode="grayscale") # load the images
                imgAD = tf.keras.utils.img_to_array(imgAD) # convert the image into an array
                imgAD = imgAD / 255.0 #normalize the image
                image_array_AD.append(imgAD)
            image_array_AD = np.array(image_array_AD) # create a numpy array
            
            for k in chunk_NC:
                path_image_NC = path_folder_images2 + k # create the path of the image
                imgNC = tf.keras.utils.load_img(path_image_NC, target_size=(height, width), color_mode="grayscale") # load the image
                imgNC = tf.keras.utils.img_to_array(imgNC) # convert the image into an array
                imgNC = imgNC / 255.0 #normalize the image
                image_array_NC.append(imgNC)
            image_array_NC = np.array(image_array_NC) # create a numpy array
            image1_list = []
            image2_list = []
            label_list= []

            for ad1_data, nc1_data, AD_index, NC_index in zip(image_array_AD, image_array_NC, indexAD, indexNC):
                ad2 = list_images1[AD_index]
                ad2 = path_folder_images1 + ad2 # create the path of the image
                ad2_data = tf.keras.utils.load_img(ad2, target_size=(height, width), color_mode="grayscale") # load the image
                ad2_data = tf.keras.utils.img_to_array(ad2_data)# convert the image into an array
                ad2_data = ad2_data / 255.0 #normalize the image

                nc2 = list_images2[NC_index]
                nc2 = path_folder_images2 + nc2 # create the path of the image
                nc2_data = tf.keras.utils.load_img(nc2, target_size=(height, width), color_mode="grayscale") # load the image
                nc2_data = tf.keras.utils.img_to_array(nc2_data)# convert the image into an array
                nc2_data = nc2_data / 255.0 #normalize the image
                #creates four pairs of images
                #AD and AD with label 1, AD and NC with label 0, 
                #NC and NC with label 1, NC and AD with label 0
                image1_list.append(ad1_data)
                image2_list.append(ad2_data)
                label_list.append(1)
                image1_list.append(ad1_data)
                image2_list.append(nc2_data)
                label_list.append(0)
                image1_list.append(nc1_data)
                image2_list.append(nc2_data)
                label_list.append(1)
                image1_list.append(nc1_data)
                image2_list.append(ad2_data)
                label_list.append(0)

            # convert lists to np array
            image1_array = np.array(image1_list)
            image2_array = np.array(image2_list)
            label_array = np.array(label_list)

            yield [image1_array, image2_array], label_array