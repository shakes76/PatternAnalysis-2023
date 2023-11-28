import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import random

batch_size = 256
image_size = 256  # We'll resize input images to this size

# read
def load_and_preprocess_image(image_path,image_size=(image_size,image_size)):
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to the desired image size
    resized_image = cv2.resize(gray_image, image_size)
    # Reshape to (64, 64, 1) for grayscale
    reshaped_image = resized_image.reshape((*image_size, 1))
    # Normalize pixel values to [0, 1]
    normalized_image = reshaped_image / 255.0
    return normalized_image

# scrape root for file paths of images and their labels
def parse_data(root_directory):
    # Initialize empty lists to store image data and labels
    x = []
    y = []

    # Traverse the directory tree
    for root, dire, _ in os.walk(root_directory):
        for d in dire:
            subdir = os.path.join(root,d)
            clas = subdir.split('/')[-1]
            #print(clas)
            if clas == 'AD':
                label = 1  # Positive class
            elif clas == 'NC':
                label = 0  # Negative class
            else:
                continue  # Skip other directories
            for _,_,filenames in os.walk(subdir):
                for fn in filenames:
                    # Get the full path of the image file            
                    image_path = os.path.join(subdir, fn)
                    #print(image_path)
                    
                    # Load and preprocess the image
                    #image_data = load_and_preprocess_image(image_path)

                    # Append the data and label to the lists
                    x.append(image_path)
                    y.append(label)
    d = list(zip(x,y))
    random.shuffle(d)
    x,y=zip(*d)
    x=list(x)
    y=list(y)
    return (x,y)

#data augs. cropping negative space
def d_augment(x):
    original_height, original_width = tf.shape(x)[0], tf.shape(x)[1]
    # Data augmentation: Random horizontal flip
    #if tf.random.uniform(()) > 0.5:
        #x = tf.image.flip_left_right(x)

    # Data augmentation: Random rotation (90, 180, or 270 degrees)
    #rotation_angle = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32) * 90
    #x = tf.image.rot90(x, k=rotation_angle // 90)
    
    row_sum = tf.reduce_sum(x,axis=0)
    col_sum = tf.reduce_sum(x,axis=1)

    minr,maxr = tf.where(tf.math.greater(row_sum,0))[0][0],tf.where(tf.math.greater(row_sum,0))[-1][0]
    minc,maxc = tf.where(tf.math.greater(col_sum,0))[0][0],tf.where(tf.math.greater(col_sum,0))[-1][0]

    x = x[minc:maxc+1,minr:maxr+1,:]

    x = tf.image.resize(x, (image_size, image_size))
    return x

# read images and format into numpy. also perform data augmentations
def tf_parse(x,y):
    def _parse(x,y):
        x = load_and_preprocess_image(x.decode())
        y = y
        x=d_augment(x)
        return x,y
    x,y=tf.numpy_function(_parse,[x,y],[tf.float32,tf.int32])
    x.set_shape([image_size,image_size,1])
    y.set_shape([])
    return x,y

# function call to transform list of file paths (X) with labels (Y) into dataset of images.
def tf_dataset(X,Y,batch_size=batch_size):
    dataset=tf.data.Dataset.from_tensor_slices((X,Y))
    dataset=dataset.map(tf_parse)
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(10)
    return dataset

def read_test_path(image_path):
    im = load_and_preprocess_image(image_path)
    im = d_augment(im)
    im = im.numpy()
    return im

