import dataset
import tensorflow as tf
from tensorflow import keras

num_epochs = 120
learning_rate = 0.001
weight_decay = 0.0001
num_classes = 2
image_size = 140
batch_size = 70

def run_model(model):
    """
    Train and evaluate a given model on a dataset.
    Args:
        model: TensorFlow model to train and evaluate.
    Returns:
        Model performance metrics during training.
    """
    # Load and preprocess the dataset
    train_images, test_images, validation_images, train_labels, test_labels, validation_labels = dataset.load_and_preprocess_data()

    # Define the optimizer with learning rate and weight decay
    optimizer = tf.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Compile the model with loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # Reduce learning rate if performance plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="accuracy",
        factor=0.2,
        patience=5,
    )

    # Train the model
    performance = model.fit(
        x=train_images,
        y=train_labels,
        callbacks=[reduce_lr],
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(validation_images, validation_labels),
    )

    # Evaluate the model on the test dataset
    _, accuracy = model.evaluate(test_images, test_labels)
    print("Test accuracy = " + str(round(accuracy, 2)))

    return performance
