import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

data_directory = '/content/drive/MyDrive/Colab Notebooks/ADNI/train'

label_mapping = {
    'AD': 0,
    'CN': 1,
}

def load_images(dir_path=data_directory, label_mapping=label_mapping):
    Images = []
    Labels = []
    width, height = 128, 128  

    for label, label_id in label_mapping.items():
        label_dir = os.path.join(dir_path, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = load_img(image_path, target_size=(width, height))
            image = img_to_array(image)

            Images.append(image)
            Labels.append(label_id)

    Images = np.array(Images, dtype=np.float32)
    Labels = np.array(Labels, dtype=np.float32)
    Images, Labels = shuffle(Images, Labels, random_state=0)

    return Images, Labels

def preprocess_data(Images, Labels):
    Images = Images / 255.0  
    Labels = to_categorical(Labels, 2)  
    return Images, Labels

def load_and_preprocess_data():
    Images, Labels = load_images()
    Images, Labels = preprocess_data(Images, Labels)
    return Images, Labels
