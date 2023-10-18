import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import create_classifier
from dataset import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras

# Load and preprocess data
Images, Labels = load_and_preprocess_data()

# Split data into training and testing sets
train_indices, test_indices = train_test_split(list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=True)


x_train = Images[train_indices]
y_train = Labels[train_indices]
x_test = Images[test_indices]
y_test = Labels[test_indices]

# Create and compile the model
model = create_classifier()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
)

# Train the model
history = model.fit(x_train, y_train, batch_size=16, epochs=30, validation_split=0.25)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)} %")

# Save the trained model
model.save('/content/drive/MyDrive/Colab Notebooks/ADNI/model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
