import numpy as np
import cv2
import tensorflow as tf
from train import load_data, create_dir, dice_coef

H = 256
W = 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.int32)
    return x

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(1)
    tf.random.set_seed(1)
    create_dir("Predictions")

    with tf.keras.utils.CustomObjectScope({'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("files/UNet.h5")

    dataset_path = "path to data here (folder both the images and masks are in)"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    dice_scores = []

    for x, y in zip(test_x, test_y):
        x = read_image(x)
        y = read_mask(y)

        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)

        dice = dice_coef(y, y_pred)
        dice_scores.append(dice)

    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Coefficient: {mean_dice:.2f}")
