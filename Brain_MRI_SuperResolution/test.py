from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('saved_models/sub_pixel_cnn_model.h5')
print(1);
test_image = cv2.imread('path_to_a_downsampled_test_image', cv2.IMREAD_GRAYSCALE)
print(1);
predicted = model.predict(np.expand_dims(np.expand_dims(test_image, axis=-1), axis=0))
print(1);
cv2.imwrite('super_resolved_image.png', predicted[0].squeeze())  # Squeeze to remove channel dimension
print(1);