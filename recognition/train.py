import torch
from dataset import *
from modules import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    total_sum = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection) / total_sum
    return dice

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU")
    else:
        print("Warning: No GPU found, using CPU")

    model_path = "files/model.h5"
    csv_path = "files/data.csv"

    dataset_path = r"C:\Users\raulm\Desktop\Uni\Sem2.2023\Patterns\ISIC-2017_Training_Data"
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    batch_size = 4
    lr = 1e-4
    num_epoch = 25
    
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)
    
    train_steps = get_steps(train_x, batch_size)
    valid_steps = get_steps(valid_x, batch_size)

    model = Unet((H, W, 3))
    metrics = [dice_coef, Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    # model.summary()
    callbacks = [
        # Save model at best val_dice_ceof
        ModelCheckpoint(model_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard()
    ]
    # Start training the model
    model.fit(
        train_dataset,
        epochs=num_epoch,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )