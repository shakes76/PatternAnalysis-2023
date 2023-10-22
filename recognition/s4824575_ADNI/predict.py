import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATA_PATH = '/Users/raghavendrasinghgulia/PatternAnalysis-2023/recognition/s4824575_ADNI/AD_NC/test'

# Load data function
def load_data():
    data = []
    labels = []

    for category in os.listdir(DATA_PATH):
        category_path = os.path.join(DATA_PATH, category)

        # Check if the current path is a directory
        if os.path.isdir(category_path):
            for img in os.listdir(category_path):
                img_path = os.path.join(category_path, img)
                image = load_img(img_path, target_size=(150, 150))  # Resize the image to match the expected input size
                image = img_to_array(image)
                data.append(image)
                labels.append(0 if category == 'AD' else 1)

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


# Load model function
def load_trained_model(model_path):
    return load_model(model_path)


# Perform prediction
def perform_prediction(model, data):
    predictions = model.predict(data)
    return predictions


# Main function for prediction
def main():
    data, labels = load_data()
    model = load_trained_model('model.h5')

    loss, accuracy = model.evaluate(data, labels)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    history = model.fit(data, labels, epochs=10, validation_split=0.2)

    # Plotting the results
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Upload a slice of an AD image
    sample_img_path = '<ADSLICE>'
    sample_image = load_img(sample_img_path, target_size=(150, 150))
    sample_image = img_to_array(sample_image)
    sample_data = np.expand_dims(sample_image, axis=0)

    # Perform prediction on the uploaded image slice
    predictions = perform_prediction(model, sample_data)
    print(f'Predictions for the AD image slice: {predictions}')

if __name__ == '__main__':
    main()
