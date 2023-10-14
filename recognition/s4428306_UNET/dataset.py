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
    return tf.data.Dataset.from_tensor_slices((data,))

#The following is modified code from:
#https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def partition(data, train_size, val_size, test_size, seed):
    #TODO: Write specification.
    data.shuffle(2596, seed)
    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size).skip(val_size)
    return train_data, val_data, test_data

#NOTE: Is some other form of normalization needed for masks?
#      e.g. Set all 255s to 1s, and everything else to 0s.

#Taken from code I wrote for prac2.
def normalize(image, mask):
    #TODO: Write specification.
    image = tf.cast(image, tf.float64) / 255.0
    mask = tf.cast(mask, tf.float64) / 255.0
    return image, mask

#Based on code from:
#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
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

def preprocessing():
    #TODO: Write specification.
    #These are the directories for the datasets.
    test_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input/"
    training_images_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/"
    training_gt_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/"
    #Load in, split, normalize and augment.
    #NOTE: test_dir is unused, there don't seem to be segmentation maps in there.
    image_data = loadDataFrom(training_images_dir, channels=3)
    mask_data = loadDataFrom(training_gt_dir, channels=1)
    isic_data = tf.data.Dataset.zip((image_data, mask_data))
    train_data, val_data, test_data = partition(isic_data, 1796, 400, 400, seed=271828)
    #NOTE: How to ensure augment happens on a batch by batch basis, but normalize doesn't?
    #      Probably want to add layers to front of model that do data augmentation, rather than doing it in
    #      preprocessing. See "tensorflow data augmentation". Would have to be layers that are only used in training.
    train_data = train_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_data = val_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #NOTE: Is any other preprocessing needed?
    #TODO: Add code that breaks train_data into batches.
    return train_data, val_data, test_data

