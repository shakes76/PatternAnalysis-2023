import tensorflow as tf
from module import transformer_model
from dataset import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Define hyperparameters
input_shape = (224, 224, 3)  # Adjust image size as needed
epochs = 3
model_save_path = 'alzheimer_model.h5'

# Load and preprocess the data
train_data_gen, test_data_gen = load_and_preprocess_data(img_height=224, img_width=224, batch_size=16)

# Build the transformer model
model = transformer_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data_gen, epochs=epochs, validation_data=test_data_gen)

# Save the trained model
model.save(model_save_path)

# Plot training history (loss and accuracy)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
