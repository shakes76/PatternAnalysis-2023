import numpy as np
import os
from PIL import Image

#These are the directories for the datasets. Not sure how they will be used.
#NOTE: These directories probably shouldn't be in here given that everything must be in functions.
#      Expecting 2595 images in the training folders (-1 for license file) (should also be -1 for attribution file).
#      Getting 2596 though, not sure why.
test_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input/"
training_images_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/"
training_gt_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/"

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
    #TODO: Convert to tensorflow tensor.(?)
    return data

"""
#Taken from code I wrote for prac2.
def normalize(item):
    #TODO: Write specification.
    image = tf.cast(item, tf.float) / 255.0
"""

