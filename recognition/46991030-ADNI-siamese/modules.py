"""
modules.py: Functions to create the SNN and classifier models

This file contains functions to create the SNN and classifier models.

This file also contains the contrastive loss function and the custom distance layer.
"""
import tensorflow as tf

import constants


@tf.function
def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Contrastive loss function.

    Args:
        y_true (tf.Tensor): The true labels.
        y_pred (tf.Tensor): The predicted labels.

    Returns:
        tf.Tensor: The contrastive loss.
    """
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(1 - y_pred, 0))
    return tf.math.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)


class DistanceLayer(tf.keras.layers.Layer):
    """
    Custom distance layer.

    This layer calculates the distance between the two encoded images.
    """

    def __init__(self, **kwargs):
        """
        Initializes the layer.
        """
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Calculates the distance between the two encoded images.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): The two encoded images.

        Returns:
            tf.Tensor: The distance between the two encoded images.
        """
        return tf.math.sqrt(
            tf.math.maximum(
                tf.math.reduce_sum(
                    tf.math.square(inputs[0] - inputs[1]), 1, keepdims=True
                ),
                tf.keras.backend.epsilon(),
            )
        )


def snn():
    """
    Creates the SNN model.

    Returns:
        tf.keras.Model: The SNN model.
    """
    input_1 = tf.keras.layers.Input(shape=constants.IMAGE_INPUT_SHAPE)
    input_2 = tf.keras.layers.Input(shape=constants.IMAGE_INPUT_SHAPE)

    twin = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64,
                (10, 10),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (7, 7),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (4, 4),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (4, 4),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(2048, activation="relu"),
        ]
    )

    encoded_1 = twin(input_1)
    encoded_2 = twin(input_2)

    distance = DistanceLayer()([encoded_1, encoded_2])

    output = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.BatchNormalization()(distance)
    )

    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)

    model.compile(optimizer="sgd", loss=loss, metrics=["accuracy"])

    return model


def snn_classifier(model: tf.keras.Model):
    """
    Creates the classifier model.

    Args:
        model (tf.keras.Model): The embedded network from the Siamese Network.

    Returns:
        tf.keras.Model: The classifier model.
    """
    input = tf.keras.layers.Input(shape=constants.IMAGE_INPUT_SHAPE)
    classifier = tf.keras.models.Sequential(
        [
            input,
            model,
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    classifier.compile(
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return classifier
