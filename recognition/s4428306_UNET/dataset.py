import tensorflow as tf
import numpy as np
import os
from PIL import Image

#NOTE: Expecting 2595 images in the training folders (-1 for license file) (should also be -1 for attribution file).
#      Getting 2596 though, not sure why.

#TODO: Change comments/spec to reference colour channels.
#Taken from code I wrote for prac2.
def loadDataFrom(directory, channels, size=128):
    """Loads the images from the given directory.

    Args:
      directory: The directory where the images are stored.

    Returns:
      NumPy array of shape (n, size, size), where n is the number
      of images in the directory.
    """
    numberOfImages = len(os.listdir(directory))
    if channels > 1:
        data = np.zeros((numberOfImages, size, size, channels))
    else:
        data = np.zeros((numberOfImages, size, size))
    for i, imageName in enumerate(os.listdir(directory)):
        #Make sure not to load in the license file.
        if imageName != "LICENSE.txt" and imageName != "ATTRIBUTION.txt":
            imagePath = os.path.join(directory, imageName)
            #NOTE: Unsure if resizing should be done in this function.
            if channels > 1:
                data[i, :, :, :] = np.asarray(Image.open(imagePath).resize((size, size)))
            else:
                data[i, :, :] = np.asarray(Image.open(imagePath).resize((size, size)))
    #NOTE: Should this be a tf dataset like this?
    return tf.data.Dataset.from_tensor_slices((data,))

#NOTE: Is some other form of normalization needed for masks?
#      e.g. Set all 255s to 1s, and everything else to 0s.

#Taken from code I wrote for prac2.
def normalize(image, mask):
    #TODO: Write specification.
    image = tf.cast(image, tf.float64) / 255.0 #NOTE: Will channels be an issue here?
    mask = tf.cast(mask, tf.float64) / 255.0
    return image, mask

#TODO: Write some other preprocessing sub-functions.

#NOTE: Which source should be referenced for this?
#TODO: Reference source.
def augment(image, mask):
    #TODO: Write specification.
    p = tf.random.uniform(())
    if p < 0.25:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    elif p < 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask

#NOTE: Function that encapsulates all preprocessing, should wind up being the only thing that's called for data.
def preprocessing():
    #TODO: Write specification.
    #These are the directories for the datasets.
    test_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input/"
    training_images_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/"
    training_gt_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/"
    #TODO: Split into test, train and validation data.
    #Load in, normalize and augment.
    image_data = loadDataFrom(training_images_dir, channels=3)
    mask_data = loadDataFrom(training_gt_dir, channels=1)
    isic_data = tf.data.Dataset.zip((image_data, mask_data))
    isic_data = isic_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #NOTE: How to ensure this happens on a batch by batch basis?
    isic_data = isic_data.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #NOTE: Is any other preprocessing needed?
    return isic_data

