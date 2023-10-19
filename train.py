import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

DATA_PATH = '/Users/raghavendrasinghgulia/PatternAnalysis-2023/AD_NC/train'

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ...

def load_data():
    data = []
    labels = []

    for category in os.listdir(DATA_PATH):
        category_path = os.path.join(DATA_PATH, category)
        
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            image = load_img(img_path, target_size=(150, 150))  # Resize the image to match the expected input size
            image = img_to_array(image)
            data.append(image)
            labels.append(0 if category == 'AD' else 1)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


 

def create_model():

  model = Sequential()

  model.add(Conv2D(32, 3, activation='relu', input_shape=(150,150,3)))
  model.add(MaxPooling2D())

  model.add(Conv2D(64, 3, activation='relu'))
  model.add(MaxPooling2D())  

  model.add(Flatten())
  model.add(Dense(128, activation='relu'))

  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

data, labels = load_data()

model = create_model()

model.fit(data, labels, epochs=10, validation_split=0.2)

model.evaluate(data, labels)

model.save('model.h5')
