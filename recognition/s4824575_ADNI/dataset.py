import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array



DATA_PATH = '/Users/raghavendrasinghgulia/PatternAnalysis-2023/recognition/s4824575_ADNI/AD_NC/train'

def load_data():
  data = []
  labels = []

  for category in os.listdir(DATA_PATH):
    category_path = os.path.join(DATA_PATH, category)
    
    for img in os.listdir(category_path):
      img_path = os.path.join(category_path, img)  
      image = load_img(img_path, target_size=(150,150))
      image = img_to_array(image)
      data.append(image)
      labels.append(0 if category == 'AD' else 1)

  data = np.array(data)
  labels = np.array(labels)

  return data, labels

if __name__ == '__main__':

  # Load data
  data, labels = load_data()

  # Normalize 
  data = data/255.

  # Split into train/test
  x_train, x_test, y_train, y_test = train_test_split(data, labels)

  # Convert to TensorFlow datasets
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))

  # Shuffle and batch datasets
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
  test_dataset = test_dataset.batch(32)
