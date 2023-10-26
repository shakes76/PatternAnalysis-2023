from modules import *
from train import *
from dataset import *
import tensorflow as tf
import numpy as np
import tensorflow_probability 
from tensorflow import keras
from keras.models import load_model
from tensorflow.image import convert_image_dtype, ssim
from matplotlib.pyplot import suptitle, imshow, subplot, axis, show, cm, title

custom_objects = {
    "VectorQuantizer": VectorQuantizer
}
with keras.utils.custom_object_scope(custom_objects):
    vqvae = load_model("VQVAE.h5")

custom_objects = {
    "PixelConvolution": PixelLayer,
    "ResidualBlock": ResidualBlock
    }
#with keras.utils.custom_object_scope(custom_objects):
#    pcnn = load_model("PCNN.h5") 

select = np.random.choice(len(x_test_scaled), 10)
images =  x_test_scaled[select]

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
    reconstructed = vqvae.predict(images)
    ssim = calculate_ssim(images, reconstructed)
    return 0

def show_codes(codebook):
    
    for i in range(len(images)):
    
        plt.subplot(1, 2, 1)
        plt.imshow(images[i].squeeze() + 0.5, cmap=plt.cm.gray)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook[i], cmap=plt.cm.gray)
        plt.title("Code")
        plt.axis("off")
        plt.show()

def generate(quantizer, priors, output_enco):
    pretrained_embeddings = quantizer.embeddings
    priors_ohe = tf.one_hot(priors.astype("int32"), vqvae.num_embeddings).numpy()
    quantized = tf.matmul(priors_ohe.astype("float32"), pretrained_embeddings, transpose_b=True)
    quantized = tf.reshape(quantized, (-1, *(output_enco.shape[1:])))

    # Generate novel images.
    decoder = vqvae.get_layer("decoder")
    generated_samples = decoder.predict(quantized)

    for i in range(10):
        plt.subplot(1, 2, 1)
        plt.imshow(priors[i], cmap = plt.cm.gray)
        plt.title("Code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(generated_samples[i].squeeze() + 0.5, cmap = plt.cm.gray)
        plt.title("Generated Sample")
        plt.axis("off")
        plt.show()

def test_pcnn():
    priors	= np.zeros(shape = (10,) + (pcnn.input_shape)[1:])
    batch, rows, cols = priors.shape

    for row in range(rows):
      for col in range(cols):
          logits = pcnn.predict(priors)
          sampler = tensorflow_probability.distributions.Categorical(logits)
          probs = sampler.sample()
          priors[:, row, col] = probs[:, row, col]

    encoder	= vqvae.get_layer("encoder")
    quantizer = vqvae.get_layer("quantizer")
    output_enco	= encoder.predict(x_test_scaled)
    codebook = quantizer.get_code_indices(output_enco.reshape(-1, output_enco.shape[-1]))
    codebook = codebook.numpy().reshape(output_enco.shape[:-1])
    show_codes(codebook)
    generate(quantizer, priors, output_enco)
    return 0

if __name__ == "__main__":
    test_vqvae()
    test_pcnn()
