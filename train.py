import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from modules import build_unet
from dataset import load_data, tf_dataset

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def calculate_dice_coefficient(model, x, y):
    y_pred = model.predict(x)
    dice = dice_coef(y, y_pred).numpy()
    return dice

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(1)
    tf.random.set_seed(1)
    create_dir("files")
    batch_size = 4
    lr = 0.0001
    num_epochs = 5
    model_path = "files/model.h5"
    csv_path = "files/data.csv"
    dataset_path = "your_dataset_directory_here (folder both the image and masks are in)"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    # To print the number of images and masks of each type of data uncomment out below
    #print(f"Train: {len(train_x)} - {len(train_y)}")
    #print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    #print(f"Test: {len(test_x)} - {len(test_y)}")
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size
    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model = build_unet((H, W, 3))
    metrics = [dice_coef]
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr), metrics=metrics)

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
    )
