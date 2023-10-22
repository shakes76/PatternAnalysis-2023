import dataset
import tensorflow as tf
from tensorflow import keras

num_epochs = 80
learning_rate = 0.005
weight_decay = 0.0001
num_classes = 2
image_size = 140
batch_size = 140


def run_model(model):
    train_images, test_images, validation_images, train_labels, test_labels, validation_labels = dataset.load_and_preprocess_data()
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="accuracy",
        factor=0.2,
        patience=5,
    )

    
    performance = model.fit(
        x=train_images,
        y=train_labels,
        callbacks=[reduce_lr],
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(validation_images, validation_labels),
    )


    _, accuracy = model.evaluate(test_images, test_labels)
    print("test accuracy= " + str(round(accuracy, 2)))

    return performance

