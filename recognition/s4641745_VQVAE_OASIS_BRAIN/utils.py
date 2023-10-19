import torchvision as tv
import os


def save_image(img, name):
    tv.utils.save_image(img, os.path.join('./assets/images', name), nrow=8)