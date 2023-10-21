import tensorflow as tf
from modules import unet_model

# Sample Configuration
EPOCHS = 10
BATCH_SIZE = 32
TRAINING_DATA_DIR = "path_to_training_images/"
TRAINING_MASK_DIR = "path_to_training_masks/"
VAL_DATA_DIR = "path_to_val_images/"
VAL_MASK_DIR = "path_to_val_masks/"
IMG_SIZE = (256, 256)

# ImageDataGenerator for data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create generators for training and validation datasets
train_image_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,  # No labels
    seed=42,
    color_mode="grayscale"
)

train_mask_generator = train_datagen.flow_from_directory(
    TRAINING_MASK_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42,
    color_mode="grayscale"
)

val_image_generator = val_datagen.flow_from_directory(
    VAL_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42,
    color_mode="grayscale"
)

val_mask_generator = val_datagen.flow_from_directory(
    VAL_MASK_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    seed=42,
    color_mode="grayscale"
)

# Combine image and mask generators
train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

# Compile and train the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=len(train_image_generator),
    validation_steps=len(val_image_generator),
    epochs=EPOCHS
)

# Save the model (optional)
model.save("unet_model.h5")
