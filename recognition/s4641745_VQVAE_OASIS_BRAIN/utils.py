import torchvision as tv
import os
from dataset import IMAGE_PATH

def save_image(img, name):
    tv.utils.save_image(img, os.path.join(IMAGE_PATH, name), nrow=8, normalize=True)