import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.sub_pixel_cnn import sub_pixel_cnn
from utils.data_loader import load_images_from_category

tf.config.run_functions_eagerly(True)

base_dir = 'AD_NC'


def train_model():
    print("Initializing model...")

    # Error handling for empty datasets
    if len(x_train) == 0 or len(y_train) == 0:
        raise ValueError("Training datasets are empty!")
    if len(x_valid) == 0 or len(y_valid) == 0:
        raise ValueError("Validation datasets are empty!")

    print(np.isnan(x_train).any())
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_valid shape:", x_valid.shape)
    print("y_valid shape:", y_valid.shape)

    input_shape = (100, 100, 1)
    model = sub_pixel_cnn(input_shape)
    sample_preds = model.predict(x_train[:5])  # take 5 samples from the training data
    print(sample_preds.shape)

    print("Compiling model...")
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', run_eagerly=True)

    print("Setting up checkpoints and early stopping...")
    checkpoint = ModelCheckpoint('saved_models/sub_pixel_cnn_best_model.h5', save_best_only=True, monitor='val_loss',
                                 mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training model...")
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=1,
                        callbacks=[checkpoint, early_stopping])

    print("Saving trained model...")
    model.save('saved_models/sub_pixel_cnn_model.keras')
    return history, model


if __name__ == "__main__":
    print("Setting base directory...")
    print("Loading AD images for training...")
    x_train_AD = load_images_from_category(base_dir, 'train', 'AD', target_size=(100, 100))
    y_train_AD = load_images_from_category(base_dir, 'train', 'AD', target_size=(400, 400))

    print("Loading NC images for training...")
    x_train_NC = load_images_from_category(base_dir, 'train', 'NC', target_size=(100, 100))
    y_train_NC = load_images_from_category(base_dir, 'train', 'NC', target_size=(400, 400))

    print("x_train_AD shape:", np.shape(x_train_AD))
    print("y_train_AD shape:", np.shape(y_train_AD))
    print("x_train_NC shape:", np.shape(x_train_NC))
    print("y_train_NC shape:", np.shape(y_train_NC))

    print("Concatenating AD and NC images...")
    x_train = np.concatenate([x_train_AD, x_train_NC], axis=0)
    y_train = np.concatenate([y_train_AD, y_train_NC], axis=0)

    print("Splitting data into training and validation sets...")
    validation_split = 0.1
    split_index = int((1 - validation_split) * len(x_train))
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    x_train = x_train[:split_index]
    y_train = y_train[:split_index]

    print("Starting training process...")
    history, trained_model = train_model()

    print("Plotting training results...")
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()
