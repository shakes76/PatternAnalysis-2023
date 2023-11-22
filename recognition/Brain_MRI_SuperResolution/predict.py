import matplotlib.pyplot as plt
import tensorflow as tf

# Constants
HEIGHT = 256
WIDTH = 256
DOWNSCALE_FACTOR = 4


# Function to calculate the Peak Signal-to-Noise Ratio (PSNR) between true and predicted images
def psnr_metric(y_true, y_pred):
    """Calculate the PSNR between y_true and y_pred."""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def predict(model, image):
    """
        Use the given model to predict the high-resolution version of an image.

        Arguments:
        - model: Trained super-resolution model
        - image: Low-resolution input image

        Returns:
        - Displays the high-resolution prediction using matplotlib
        """
    # Expanding the dimensions of the input image for the model prediction
    input = tf.expand_dims(image, axis=0)
    # Getting the model prediction
    output = model.predict(input)
    # Displaying the output image
    plt.imshow(output[0] / 255.0)

    plt.title("Upscaled Image")
    plt.axis('off')
    plt.show()


def displayPredictions(model, test_image_path):
    """
       Load a test image, display its original and low-resolution versions,
       and then use the model to predict and display its high-resolution version.

       Arguments:
       - model: Trained super-resolution model
       - test_image_path: Path to the test image
       """

    # Load the test image
    image = tf.io.read_file(test_image_path)
    image = tf.io.decode_jpeg(image, channels=1)

    # Resize the image to original dimensions and downscale it to get the low-resolution version
    original = tf.image.resize(image, [HEIGHT, WIDTH])
    downscaled = tf.image.resize(original, [100, 100], method=tf.image.ResizeMethod.BILINEAR)

    # Plot and display the original and downscaled images side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original.numpy().squeeze(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(downscaled.numpy().squeeze(), cmap='gray')
    plt.title("Downscaled Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Predict the high-resolution version of the downscaled image
    predict(model, downscaled)


# Load model and image
model_path = 'saved_models/sub_pixel_cnn_model.h5'
model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "LeakyReLU": tf.keras.layers.LeakyReLU,
        "psnr_metric": psnr_metric
    }
)
# Image used fot testing
test_image_path = 'AD_NC/test/AD/390461_93.jpeg'

# Display the test image predictions using the model
displayPredictions(model, test_image_path)
