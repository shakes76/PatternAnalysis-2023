import os
import cv2
import numpy as np


def resize_images(folder_path, target_size=(256, 256), extensions=None):
    """
    :param folder_path: path to folder containing images
    :param target_size: size of image to resize
    :param extensions: list of file extensions to consider as images
    :return: numpy array of resized images
    """
    if extensions is None:
        extensions = [".jpg", ".png"] # inputs in jpg and grand truth in png
    resized_images = []
    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in extensions):  # only process image files
            image_path = os.path.join(folder_path, filename)

            # Read and resize the image
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, target_size)

            resized_images.append(image_resized)

    return np.array(resized_images)

