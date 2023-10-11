import numpy as np
import cv2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from AD_NC.dataset import load_dataset, downsample_images
from models.sub_pixel_cnn import sub_pixel_cnn


def resize_images(images, target_dim):
    return [cv2.resize(image, target_dim, interpolation=cv2.INTER_CUBIC) for image in images]


def prepare_data():
    originals = load_dataset('AD_NC/train/AD') + load_dataset('AD_NC/train/NC')
    downsampled = downsample_images(originals)

    print("Original images count:", len(originals))
    print("Sample original image shape:", originals[0].shape)
    print("Downsampled images count:", len(downsampled))
    print("Sample downsampled image shape:", downsampled[0].shape)

    originals_resized = resize_images(originals, (128, 120))
    print("Resized originals count:", len(originals_resized))
    print("Sample resized original image shape:", originals_resized[0].shape)

    X_train = np.array(downsampled).astype(np.float32) / 255.0  # Normalizing to [0,1]
    Y_train = np.array(originals_resized).astype(np.float32) / 255.0  # Normalizing to [0,1]

    return X_train, Y_train


def train_model(X_train, Y_train):
    input_shape = X_train[0].shape + (1,)  # Adding the channel dimension
    model = sub_pixel_cnn(input_shape)
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    checkpoint = ModelCheckpoint('saved_models/sub_pixel_cnn_best_model.h5', save_best_only=True, monitor='val_loss',
                                 mode='min')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Training data shape:", X_train.shape)
    print("Target data shape:", Y_train.shape)

    model.fit(X_train, Y_train, validation_split=0.1, epochs=50, batch_size=16, callbacks=[checkpoint, early_stopping])

    # Save the final model
    model.save('saved_models/sub_pixel_cnn_model.keras')


if __name__ == "__main__":
    X_train, Y_train = prepare_data()
    train_model(X_train, Y_train)
