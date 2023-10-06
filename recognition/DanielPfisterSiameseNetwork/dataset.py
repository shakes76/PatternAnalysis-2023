import os
import numpy as np
import tensorflow as tf

#define the function to read all the image names from folder 
#und count the image number
def image_list(path_folder_images):
    list_image = sorted(os.listdir(path_folder_images))
    image_count = len(list_image)
    print("Number of images: " + str(image_count))
    
    return list_image, image_count

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