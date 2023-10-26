import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta 
        )
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        indices = tf.argmin(distances, axis=1)
        return indices

def get_encoder(latent_dim):
    inputs = keras.Input(shape=(80, 80, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x1 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x2 = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x1)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x2)
    return keras.Model(inputs, encoder_outputs, name="encoder")

def get_decoder(latent_dim):
    inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(inputs)
    x1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x1)   
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x2)
    return keras.Model(inputs, decoder_outputs, name="decoder")

def vqvae(lat, emb, beta):
    vect_q = VectorQuantizer(emb, lat, beta, name = "quantizer")
    encoder	= get_encoder(lat)
    decoder	= get_decoder(lat)
    input	= keras.Input(shape = (80,80,1))
    encoder_out	= encoder(input)
    quantized	= vect_q(encoder_out)
    reconstructed	= decoder(quantized)

    return keras.Model(input, reconstructed, name = "vqvae")

class VQVAE_MODEL(tf.keras.Model):
    def __init__(self, variance, latent_dim, num_embeddings, beta, **kwargs):
        super(VQVAE_MODEL, self).__init__(**kwargs)
        self.variance  = variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.vqvae = vqvae(self.latent_dim, self.num_embeddings, beta)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ] 
    
    def train_step(self, x):
        
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
class PixelLayer(layers.Layer):
    
    def __init__(self, mask_type, **kwargs):
        super(PixelLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel)
        self.mask[: kernel[0] // 2, ...] = 1.0
        self.mask[kernel[0] // 2, : kernel[1] // 2, ...] = 1.0
        
        if self.mask_type == "B":
            self.mask[kernel[0] // 2, kernel[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)
    
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
    
def pcnn(vqvae_model, output_enco):
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    pixelcnn_input_shape = output_enco.shape[1:-1]
    print(f"Input shape of the PixelCNN: {pixelcnn_input_shape}")
    
    pixelcnn_inputs = keras.Input(shape=pixelcnn_input_shape, dtype=tf.int32)
    ohe = tf.one_hot(pixelcnn_inputs, vqvae_model.num_embeddings)
    x = PixelLayer(mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same")(ohe)

    for _ in range(num_residual_blocks):
        x = ResidualBlock(filters=128)(x)

    for _ in range(num_pixelcnn_layers):
        x = PixelLayer(
            mask_type="B",
            filters=128,
            kernel_size=1,
            strides=1,
            activation="relu",
            padding="valid",
        )(x)  
    outputs = keras.layers.Conv2D(filters=vqvae_model.num_embeddings, kernel_size=1, strides=1, padding="valid")(x)

    pcnn = keras.Model(pixelcnn_inputs, outputs, name="pixel_cnn")
    
    return pcnn