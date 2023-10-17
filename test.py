import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained super-resolution model
model = load_model('super_resolution_model.h5')  # Replace with the path to your saved model

# Load and preprocess your test data
# Replace 'test_data_dir' with the path to your test data
# You may need to adjust the input size to match your model's requirements
test_image = Image.open('test_image.jpg')  # Replace with your test image file
test_image = test_image.resize((128, 120))  # Adjust to match your model's input size

# Convert the image to a NumPy array
test_image = np.array(test_image) / 255.0  # Normalize pixel values to [0, 1]
test_image = np.expand_dims(test_image, axis=0)  # Add a batch dimension

# Perform super-resolution (upscale)
super_resolution_result = model.predict(test_image)

# You can now save or display the super-resolved image
super_resolution_result = super_resolution_result[0] * 255.0  # Denormalize pixel values
super_resolution_result = np.clip(super_resolution_result, 0, 255).astype(np.uint8)

super_resolution_image = Image.fromarray(super_resolution_result)
super_resolution_image.save('super_resolved_image.jpg')  # Save the super-resolved image
super_resolution_image.show()  # Display the super-resolved image
