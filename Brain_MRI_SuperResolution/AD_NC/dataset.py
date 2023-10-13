import os
import cv2
import numpy as np

def load_dataset(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def downsample_images(images, factor=4):
    return [cv2.resize(img, (img.shape[1] // factor, img.shape[0] // factor), interpolation=cv2.INTER_LINEAR) for img in images]
