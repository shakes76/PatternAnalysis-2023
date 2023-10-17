import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from PIL import Image
import numpy as np
import os

# Path to the directory containing your training and test data
train_data_directory = 'AD_NC/train'
test_data_directory = 'AD_NC/test'

# Load and preprocess training data (X_train)
X_train = []
for class_name in ["AD", "NC"]:
    class_dir = os.path.join(train_data_directory, class_name)
    class_image_paths = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]

    for train_image_path in class_image_paths:
        train_image = Image.open(train_image_path)
        desired_width, desired_height = 256, 256  # Adjust dimensions as needed
        train_image = train_image.resize((desired_width, desired_height))
        train_image = np.array(train_image, dtype='float32') / 255.0  # Convert to float32 and normalize
        X_train.append(train_image)

X_train = np.array(X_train)

# Load and preprocess test data (X_test)
X_test = []
for class_name in ["AD", "NC"]:
    class_dir = os.path.join(test_data_directory, class_name)
    class_image_paths = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]

    for test_image_path in class_image_paths:
        test_image = Image.open(test_image_path)
        desired_width, desired_height = 256, 256  # Adjust dimensions as needed
        test_image = test_image.resize((desired_width, desired_height))
        test_image = np.array(test_image, dtype='float32') / 255.0  # Convert to float32 and normalize
        X_test.append(test_image)

X_test = np.array(X_test)

# Define and compile your super-resolution model
model = models.Sequential()
# Add layers to your model based on your chosen architecture
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Training parameters
num_epochs = 5  # Adjust the number of epochs based on your dataset and convergence
batch_size = 16  # Adjust the batch size according to your available memory

# Train the model
model.fit(X_train, X_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model on test data
test_loss = model.evaluate(X_test, X_test)
print(f'Test Loss: {test_loss}')

# Save the trained model
model.save('super_resolution_model.h5')
