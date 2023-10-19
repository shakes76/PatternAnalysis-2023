import tensorflow as tf
from dataset import load_data

# Example data loading and preprocessing logic
# Replace this with your actual data loading and preprocessing logic
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize the data and perform any necessary preprocessing
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    # Load the model
    model = tf.keras.models.load_model('train.py')  # Replace 'my_model.h5' with the path to your saved model

    # Load and preprocess the data for prediction
    _, (x_test, y_test) = load_data()
    x_sample = x_test[0]  # You can choose any sample from the test set
    x_sample = x_sample.reshape(1, 28, 28)  # Reshape to match the input shape of the model

    # Perform the prediction
    predictions = model.predict(x_sample)
    predicted_label = tf.argmax(predictions, axis=1).numpy()[0]

    print(f"Predicted label: {predicted_label}")
