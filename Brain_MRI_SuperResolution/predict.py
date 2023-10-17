import numpy as np
import cv2
from tensorflow.keras.models import load_model
from models import sub_pixel_cnn

print("[DEBUG] Starting the program.")

test_image_path = 'AD_NC/test/AD/388206_87.jpeg'
print(f"[DEBUG] Loading image from path: {test_image_path}")
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

if test_image is None:
    print("Error loading the image!")
    exit()

print("[DEBUG] Image loaded successfully.")
print(f"[DEBUG] Image shape: {test_image.shape}")

model_path = 'saved_models/sub_pixel_cnn_model.h5'
print(f"[DEBUG] Loading model from path: {model_path}")
model = load_model(model_path)
print("[DEBUG] Model loaded successfully.")

# Resize the image to the expected input size for the model
print("[DEBUG] Resizing the image for model input.")
test_image_downsampled = cv2.resize(test_image, (64, 60), interpolation=cv2.INTER_CUBIC)
print(f"[DEBUG] Resized image shape: {test_image_downsampled.shape}")

input_data = np.expand_dims(np.expand_dims(test_image_downsampled, axis=-1), axis=0)
print("[DEBUG] Preparing image data for prediction.")
print(f"[DEBUG] Input data shape: {input_data.shape}")

print("[DEBUG] Running the prediction.")
predicted_image = model.predict(input_data)
print("[DEBUG] Prediction completed.")
print(f"[DEBUG] Predicted image shape: {predicted_image[0].shape}")

# Rescale values from 0-1 to 0-255 and convert to uint8 type for displaying/saving with OpenCV
predicted_image_rescaled = (predicted_image[0] * 255).astype(np.uint8)
print("[DEBUG] Rescaled predicted image for visualization.")

# # Display the image
# print("[DEBUG] Displaying the predicted image.")
# cv2.imshow('Predicted Image', predicted_image_rescaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Display the images side by side for comparison
# Ensure all images have the same height before concatenation
height = test_image.shape[0]
width_ratio_downsampled = test_image_downsampled.shape[1] / test_image_downsampled.shape[0]
width_ratio_predicted = predicted_image_rescaled.shape[1] / predicted_image_rescaled.shape[0]


test_image_downsampled_resized = cv2.resize(test_image_downsampled, (test_image_downsampled.shape[1] * height // test_image_downsampled.shape[0], height))
predicted_image_rescaled_resized = cv2.resize(predicted_image_rescaled, (predicted_image_rescaled.shape[1] * height // predicted_image_rescaled.shape[0], height))


concatenated_output = np.hstack((test_image, test_image_downsampled_resized, predicted_image_rescaled_resized))
cv2.imshow('Original | Downsampled | Predicted', concatenated_output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Optionally, save the image
# cv2.imwrite('path_to_save_predicted_image.png', predicted_image_rescaled)
print("[DEBUG] Program finished successfully.")
