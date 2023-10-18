import tensorflow as tf

import constants


@tf.function
def loss(y_true, y_pred):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(1 - y_pred, 0))
    return tf.math.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)


class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.sqrt(
            tf.math.maximum(
                tf.math.reduce_sum(
                    tf.math.square(inputs[0] - inputs[1]), 1, keepdims=True
                ),
                tf.keras.backend.epsilon(),
            )
        )


def snn():
    input_1 = tf.keras.layers.Input(shape=constants.IMAGE_INPUT_SHAPE)
    input_2 = tf.keras.layers.Input(shape=constants.IMAGE_INPUT_SHAPE)

    twin = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64,
                (10, 10),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (7, 7),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                128,
                (4, 4),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                256,
                (4, 4),
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                kernel_initializer="he_uniform",
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
    input = tf.keras.layers.Input(shape=constants.IMAGE_INPUT_SHAPE)
    classifier = tf.keras.models.Sequential(
        [
            input,
            model,
            tf.keras.layers.Dense(1024, activation="relu"),
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
