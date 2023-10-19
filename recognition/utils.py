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


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    total_sum = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection) / total_sum
    return dice


class DiceThresholdStop(Callback):
    def __init__(self, threshold):
        super(DiceThresholdStop, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_dice = logs.get("val_dice_coef")
        if current_dice is not None and current_dice >= self.threshold:
            print(f"\nReached dice coefficient threshold {self.threshold} training stoped")
            self.model.stop_training = True