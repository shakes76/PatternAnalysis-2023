import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

H = 256
W = 256

def read_image(path):
    path = path.decode()
    # (H, W, 3) as RBG 3 chanels
    x = cv2.imread(path, cv2.IMREAD_COLOR)  
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    # (256, 256, 3)
    return x  


def read_image_predict(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)


def read_mask(path):
    path = path.decode()
    # (H, W) no channels present
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    x = cv2.resize(x, (W, H))
    x = x/255.0
    # (256, 256)
    x = x.astype(np.float32) 
    # Add chanel for gs (256, 256, 1)            
    x = np.expand_dims(x, axis=-1)
    return x

def read_mask_predict(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)                    ## (256, 256)
    return ori_x, x

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    total_sum = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection) / total_sum
    return dice

def dice_coef_predict(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    total_sum = tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32))
    dice = (2.0 * intersection) / total_sum
    return dice

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred*255], axis=1)
    print(f"Saving image to: {save_image_path}")
    cv2.imwrite(save_image_path, cat_images)


class DiceThresholdStop(Callback):
    def __init__(self, threshold):
        super(DiceThresholdStop, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_dice = logs.get("val_dice_coef")
        if current_dice is not None and current_dice >= self.threshold:
            print(f"\nReached dice coefficient threshold {self.threshold} training stoped")
            self.model.stop_training = True