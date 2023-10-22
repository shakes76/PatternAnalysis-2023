import tensorflow as tf
from module import build_vision_transformer
from dataset import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from parameters import *

# Define hyperparameters
input_shape = (128, 128, 3)  # Adjust image size 
epochs = 3
model_save_path = 'alzheimer_model.h5'

# Load and preprocess the data
train_data_gen, test_data_gen = load_and_preprocess_data(img_height=128, img_width=128, batch_size=16)

# Build the transformer model
model = build_vision_transformer(
        INPUT_SHAPE,
        IMAGE_SIZE,
        PATCH_SIZE,
        NUM_PATCHES,
        ATTENTION_HEADS,
        PROJECTION_DIM,
        HIDDEN_UNITS,
        DROPOUT_RATE,
        TRANSFORMER_LAYERS,
        MLP_HEAD_UNITS
    )

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
