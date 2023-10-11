import tensorflow as tf
import tensorflow.keras as k
import keras.layers as kl
import keras.backend as kb
from keras.models import Model

def contrastive_loss(y, y_pred):
    """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
            
    """
    ### Reference: https://keras.io/examples/vision/siamese_contrastive/

    square_pred = tf.math.square(y_pred)
    margin = tf.math.square(tf.math.maximum(1 - (y_pred) , 0))
    return tf.math.reduce_mean((1 - y) * square_pred + (y) * margin)