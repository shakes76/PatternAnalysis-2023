import matplotlib.pyplot as plt
import tensorflow as tf

# Constants
HEIGHT = 256
WIDTH = 256
DOWNSCALE_FACTOR = 4

import tensorflow as tf

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def predict(model, image):
    """Use the model to predict the image from the lowres image and plot results"""
    input = tf.expand_dims(image, axis=0)
    output = model.predict(input)
    plt.imshow(output[0] / 255.0)

    # plt.imshow(output[0] / 255.0, cmap='gray')

    plt.title("Upscaled Image")
    plt.axis('off')
    plt.show()


def displayPredictions(model, test_image_path):
    """Function used to display the Original image, the low resolution image
        and the upscaled image the model has predicted"""

    # Load the image
    image = tf.io.read_file(test_image_path)
    image = tf.io.decode_jpeg(image, channels=1)
    original = tf.image.resize(image, [HEIGHT, WIDTH])
    downscaled = tf.image.resize(original, [100, 100], method=tf.image.ResizeMethod.BILINEAR)

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
test_image_path = 'AD_NC/test/AD/388206_87.jpeg'

displayPredictions(model, test_image_path)
