import os
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from modules import build_unet
from dataset import load_data, tf_dataset

H = 256
W = 256

def create_dir(path):
    """
    Create a directory if it does not exist.

    Parameters:
    - path (str): The path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def dice_coef(y_true, y_pred):
    """
    Calculate the Dice coefficient.

    Parameters:
    - y_true (tf.Tensor): The true labels.
    - y_pred (tf.Tensor): The predicted labels.

    Returns:
    - tf.Tensor: The Dice coefficient.
    """
    smooth = 1e-5

    # Convert y_true and y_pred to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    # Ensure that the dice coefficient is of dtype float32
    return tf.cast(dice, tf.float32)

def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss.

    Parameters:
    - y_true (tf.Tensor): The true labels.
    - y_pred (tf.Tensor): The predicted labels.

    Returns:
    - tf.Tensor: The Dice loss.
    """
    return 1.0 - dice_coef(y_true, y_pred)

def calculate_dice_coefficient(model, x, y):
    """
    Calculate the Dice coefficient for a given model and input data.

    Parameters:
    - model: The neural network model.
    - x: Input data.
    - y: True labels.

    Returns:
    - float: The calculated Dice coefficient.
    """
    y_pred = model.predict(x)
    dice = dice_coef(y, y_pred).numpy()
    return dice

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(1)
    tf.random.set_seed(1)

    # Check if CUDA (GPU support) is available in PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    create_dir("files")
    batch_size = 4
    lr = 0.0001
    num_epochs = 11
    model_path = "files/model.h5"

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data()
    # To print the number of images and masks of each type of data uncomment out below
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
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

    # Define a ModelCheckpoint callback
    checkpoint = ModelCheckpoint(model_path, monitor='val_dice_coef', save_best_only=True, mode='max', verbose=1)

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=[checkpoint]
    )
