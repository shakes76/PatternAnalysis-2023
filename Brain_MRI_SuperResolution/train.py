import numpy as np
import cv2
from AD_NC.dataset import load_dataset, downsample_images
from models.sub_pixel_cnn import sub_pixel_cnn
from tensorflow.keras.optimizers import Adam

def resize_images(images, target_dim):
    return [cv2.resize(image, target_dim, interpolation=cv2.INTER_CUBIC) for image in images]

# Load datasets
originals = load_dataset('AD_NC/train/AD') + load_dataset('AD_NC/train/NC')
downsampled = downsample_images(originals)

print("Original images count:", len(originals))
print("Sample original image shape:", originals[0].shape)
print("Downsampled images count:", len(downsampled))
print("Sample downsampled image shape:", downsampled[0].shape)

# Resize originals to match the expected model output size
originals_resized = resize_images(originals, (120, 128))
print("Resized originals count:", len(originals_resized))
print("Sample resized original image shape:", originals_resized[0].shape)

# Model
input_shape = downsampled[0].shape + (1,)
model = sub_pixel_cnn(input_shape)
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# Optimizer and compilation
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Training data preparation
X_train = np.array(downsampled).reshape(-1, *input_shape)
Y_train = np.array(originals_resized).reshape(-1, 120, 128, 1)

print("Training data shape:", X_train.shape)
print("Target data shape:", Y_train.shape)

# Training
model.fit(X_train, Y_train, epochs=50, batch_size=16)

# Save model
model.save('saved_models/sub_pixel_cnn_model.h5')
