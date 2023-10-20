# modules.py containing the source code of the components of the model.

import tensorflow as tf
import tensorflow.keras as k
import keras.layers as kl
import keras.backend as kb
from keras.models import Model



def euclidean_distance(feature1, feature2):

    # compute the sum of squared distances between the vectors
    sum_square = tf.math.reduce_sum(tf.math.square(feature1 - feature2), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def contrastive_loss(y, y_pred):
    ### Reference: https://keras.io/examples/vision/siamese_contrastive/ 
    """Calculates the contrastive loss.

        Arguments:
            y: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
            
    """

    square_pred = tf.math.square(y_pred)
    margin = tf.math.square(tf.math.maximum(1 - (y_pred) , 0))
    return tf.math.reduce_mean((1 - y) * square_pred + (y) * margin)


def siamese_network(height, width):

    # CNN model
    cnn = k.Sequential(layers=[
            kl.Conv2D(32, (10, 10), activation='relu', input_shape=(height, width, 1), kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.MaxPooling2D(),
            kl.Conv2D(64, (7, 7), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.MaxPooling2D(),
            kl.Conv2D(64, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.MaxPooling2D(),
            kl.BatchNormalization(),
            kl.Conv2D(128, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.Flatten(),
            # kl.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # kl.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ], name='cnn')

    image1 = tf.keras.Input((height, width, 1))
    image2 = tf.keras.Input((height, width, 1))

    # Embedding network
    feature1 = cnn(image1)
    feature2 = cnn(image2)

    # Calculate the euclidean distance between feature1 and feature2
    distance = euclidean_distance(feature1, feature2)

    normal_layer = kl.BatchNormalization()(distance)
    output_layer = kl.Dense(1, activation='sigmoid')(normal_layer)
    model = tf.keras.Model(inputs=[image1, image2], outputs=output_layer)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(loss=contrastive_loss, metrics=['accuracy'], optimizer=opt)
    return model


def cnn (height, width):
    cnn = k.Sequential(layers=[
            kl.Conv2D(32, (10, 10), activation='relu', input_shape=(height, width, 1), kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.MaxPooling2D(),
            kl.Conv2D(64, (7, 7), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.MaxPooling2D(),
            kl.Conv2D(64, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.MaxPooling2D(),
            kl.BatchNormalization(),
            kl.Conv2D(128, (4, 4), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            kl.Flatten(),
            # kl.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # kl.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ], name='cnn')


def classification_model (cnn) :

    image = kl.Input((128,128,1))

    feature = cnn(image)
    feature = kl.BatchNormalization()(feature)

    out = kl.Dense(units = 1, activation= 'sigmoid')(feature)
    classifier = tf.keras.Model([image], out)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001) 

    classifier.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    return classifier

