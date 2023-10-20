# imports
import datasets
import tensorflow as tf
from tensorflow import keras

# hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 100
img_size = 128
batch_size = 128
num_classes = 2


def run_model(model):

    # import data
    x_train, y_train, x_test, y_test, x_val, y_val = datasets.create_data()
    # define optimiser
    optimiser = tf.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    # define model
    model.compile(
        optimizer=optimiser,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # reduce learning rate as model progresses
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="accuracy",  # reduces learning rate based on accuracy
        factor=0.2,  # new_lr = lr * factor
        patience=3  # only reduce if plateaus for 5 straight epochs
    )
    # train model
    history = model.fit(
        x=x_train,  # images
        y=y_train,  # labels
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[reduce_lr],  # reduction method
        validation_data=(x_val, y_val),  # validation data
    )
    # evaluate model
    _, accuracy = model.evaluate(x_test, y_test)
    print("test accuracy = " + str(round(accuracy * 100, 2)))

    return history
