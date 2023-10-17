# imports
import datasets
import tensorflow as tf
from tensorflow import keras

learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 3
img_size = 128
batch_size = 128
num_classes = 2


def run_model(model):

    x_train, y_train, x_test, y_test, x_val, y_val = datasets.create_data()

    optimiser = tf.optimizers.Adam(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimiser,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # reduce learning rate as model progresses
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="accuracy", factor=0.1, patience=3
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[reduce_lr],
        validation_data=(x_val, y_val),
    )

    _, accuracy =model.evaluate(x_test, y_test)
    print("test accuracy = " + str(round(accuracy * 100, 2)))

    return history




