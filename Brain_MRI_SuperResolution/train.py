import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model.modules import sub_pixel_cnn
from utils.dataset import load_images_from_category

# Configuring TensorFlow to run functions eagerly for debugging and development
tf.config.run_functions_eagerly(True)

# Base directory for dataset
base_dir = 'AD_NC'


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# Function to get a subset of a dataset
def subset_data(x, y, fraction=0.1):
    """
    Get a subset of the dataset
    """
    subset_size = int(len(x) * fraction)
    return x[:subset_size], y[:subset_size]


# Function to train the super-resolution model
def train_model():

    # Error handling for empty datasets
    if len(x_train) == 0 or len(y_train) == 0:
        raise ValueError("Training datasets are empty!")
    if len(x_valid) == 0 or len(y_valid) == 0:
        raise ValueError("Validation datasets are empty!")

    # Define the model structure
    input_shape = (100, 100, 1)
    model = sub_pixel_cnn(input_shape)
    sample_preds = model.predict(x_train[:5])  # take 5 samples from the training data
    print(sample_preds.shape)

    print("Compiling model...")
    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[psnr_metric], run_eagerly=True)

    checkpoint = ModelCheckpoint('saved_models/sub_pixel_cnn_best_model.h5', save_best_only=True, monitor='val_loss',
                                 mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=100,
                        callbacks=[checkpoint, early_stopping])

    model.save('saved_models/sub_pixel_cnn_model.h5')
    return history, model


# Main execution starts
if __name__ == "__main__":
    print("Setting base directory...")
    x_train_AD = load_images_from_category(base_dir, 'train', 'AD', target_size=(100, 100))
    y_train_AD = load_images_from_category(base_dir, 'train', 'AD', target_size=(400, 400))
    x_train_AD, y_train_AD = subset_data(x_train_AD, y_train_AD, fraction=0.0001)

    print("Loading NC images for training...")
    x_train_NC = load_images_from_category(base_dir, 'train', 'NC', target_size=(100, 100))
    y_train_NC = load_images_from_category(base_dir, 'train', 'NC', target_size=(400, 400))
    x_train_NC, y_train_NC = subset_data(x_train_NC, y_train_NC, fraction=0.0001)

    print("Concatenating AD and NC images...")
    # Concatenating AD and NC images to create the training dataset
    x_train = np.concatenate([x_train_AD, x_train_NC], axis=0)
    y_train = np.concatenate([y_train_AD, y_train_NC], axis=0)

    print("Splitting data into training and validation sets...")
    # Splitting data into training and validation sets
    validation_split = 0.1
    split_index = int((1 - validation_split) * len(x_train))
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    x_train = x_train[:split_index]
    y_train = y_train[:split_index]


    # Begin the training process
    history, trained_model = train_model()
    print(history.history)

    import matplotlib.pyplot as plt

    # Visualize the training results
    print("Plotting training results...")
    plt.ylim(0, 0.05)  # adjust as needed based on your loss range

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig("Images/training_plot.png")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr_metric'], label='Training PSNR', color='blue')
    plt.plot(history.history['val_psnr_metric'], label='Validation PSNR', color='red')
    plt.title('Training vs Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR Value')
    plt.legend()
    plt.savefig("Images/PSNR.png")

plt.tight_layout()
plt.show()
