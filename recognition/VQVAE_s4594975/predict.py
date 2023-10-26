from modules import *
from train import *
from dataset import *
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.image import convert_image_dtype, ssim
from matplotlib.pyplot import suptitle, imshow, subplot, axis, show, cm, title

custom_objects = {
    "VectorQuantizer": VectorQuantizer
}
with keras.utils.custom_object_scope(custom_objects):
    vqvae = load_model("/Users/pc/Documents/COMP3710/VQVAE.h5")
#pcnn = load_model("/Users/pc/Documents/COMP3710/PCNN.h5", custom_objects = {"PixelConvolution": PixelLayer, "ResidualBlock": ResidualBlock})


def calculate_ssim(images, recons):
    sum = 0
    for test_image, reconstructed_image in zip(images, recons):
        test = convert_image_dtype(test_image, tf.float32)
        reconstructed_image = convert_image_dtype(reconstructed_image, tf.float32)
        ssim = tf.image.ssim(test, reconstructed_image, 1.0)
        sum = sum + ssim
        plt.suptitle("SSIM: %.2f" %ssim)   
        plt.subplot(1, 2, 1)

        plt.imshow(tf.squeeze(test_image) + 0.5, cmap=plt.cm.gray)
        plt.title("Test_image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(tf.squeeze(reconstructed_image) + 0.5, cmap=plt.cm.gray)
        plt.title("Reconstructed")
        plt.axis("off")
        plt.show()
    print(sum / len(recons))

def test_vqvae():
    select = np.random.choice(len(x_test_scaled), 1000)
    images =  x_test_scaled[select]
    reconstructed = vqvae.predict(images)
    ssim = calculate_ssim(images, reconstructed)
    return 0

def test_pcnn():
    return 0

if __name__ == "__main__":
    test_vqvae()
    test_pcnn()
