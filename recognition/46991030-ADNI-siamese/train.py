import tensorflow as tf

import constants
import dataset
import modules

(
    train_ds,
    validate_ds,
    test_ds,
    class_train_ds,
    class_validate_ds,
    class_test_ds,
) = dataset.load_dataset(constants.DATASET_PATH)

print("Creating SNN model")
model = modules.snn()

print("Training SNN model")
model.fit(
    train_ds,
    epochs=10,
    validation_data=validate_ds,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=0.0001
        )
    ],
)

print("Testing SNN model")
model.evaluate(test_ds, verbose=1)

print("Saving SNN model")
model.save("models/snn", save_format="tf")

twin = model.get_layer("sequential")
twin.trainable = False

classifier = modules.snn_classifier(twin)

classifier.fit(
    class_train_ds,
    epochs=30,
    validation_data=class_validate_ds,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
        )
    ],
)

classifier.evaluate(class_test_ds, verbose=1)

classifier.save("models/classifier", save_format="tf")
