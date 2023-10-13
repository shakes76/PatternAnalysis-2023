import numpy as np
import cv2
from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.sub_pixel_cnn import sub_pixel_cnn
from AD_NC.data_loader import load_data

# Load datasets using the load_data function from data_loader.py
train_ds, valid_ds = load_data()
img_height = 100
img_width = 100
def resize_image(image, target_dim):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(target_dim[1] * aspect_ratio)
    new_height = target_dim[0]

    if new_width > target_dim[0]:
        new_width = target_dim[0]
        new_height = int(new_width / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    delta_w = max(target_dim[1] - new_width, 0)
    delta_h = max(target_dim[0] - new_height, 0)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    final_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return final_image



def resize_images(images, target_dim):
    return [resize_image(img, target_dim) for img in images]


def train_model():
    input_shape = (img_height, img_width, 1)  # Assuming images are RGB. Change 3 to 1 for grayscale.
    model = sub_pixel_cnn(input_shape)

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    checkpoint = ModelCheckpoint('saved_models/sub_pixel_cnn_best_model.h5', save_best_only=True, monitor='val_loss',
                                 mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=valid_ds, epochs=1,
                        callbacks=[checkpoint, early_stopping])

    model.save('saved_models/sub_pixel_cnn_model.keras')
    return history, model


if __name__ == "__main__":
    history, trained_model = train_model()

    import matplotlib.pyplot as plt

    # Plotting Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()
 #
 # for batch in val_dataset.take(1):
 #        for low_res, high_res_real in batch:
 #            high_res_pred = trained_model.predict(low_res)
 #            plot_results(high_res_real[0], "HR_real", "High Resolution - Real")
 #            plot_results(high_res_pred[0], "HR_pred", "High Resolution - Predicted")
