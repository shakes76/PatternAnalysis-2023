import os
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Data paths
DATA_DIR = '/Users/raghavendrasinghgulia/PatternAnalysis-2023/recognition/s4824575_ADNI/AD_NC/train'
DATA_PATH_AD = os.path.join(DATA_DIR, 'AD')  
DATA_PATH_NC = os.path.join(DATA_DIR, 'NC')

# Load and preprocess data
def load_data():
  data = []
  labels = []

  for category in ['AD', 'NC']:
    path = DATA_PATH_AD if category=='AD' else DATA_PATH_NC
    for img in os.listdir(path):
      img_path = os.path.join(path, img)  
      image = load_img(img_path, target_size=(150,150))
      image = img_to_array(image)
      data.append(image)
      labels.append(0 if category=='AD' else 1)

  return np.array(data), np.array(labels)

data, labels = load_data()

# Create CNN model
def create_model():
  model = Sequential()
  model.add(Conv2D(32, 3, activation='relu', input_shape=(150,150,3))) 
  model.add(MaxPooling2D())
  model.add(Conv2D(64, 3, activation='relu'))
  model.add(MaxPooling2D()) 
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = create_model()

# Train model  
checkpoint = ModelCheckpoint('model-{epoch:02d}.h5', 
                            monitor='val_loss', 
                            verbose=0, 
                            save_best_only=True,
                            mode='min')

history = model.fit(data, labels, 
                    epochs=10,
                    validation_split=0.2,
                    callbacks=[checkpoint])

# Save the final model
model.save('model.h5') 

# Plot results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png')

plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')  
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
plt.show()