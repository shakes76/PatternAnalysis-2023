"""
Example usage of the trained model
"""
import keras
import tensorflow_probability as tfp
from train import *
from keras import layers

TRAINING_VQVAE = False
TRAINING_PIXELCNN = False

vqvae = VQVAE(tr_var=var_pixel_train, num_encoded=NUM_EMBEDDINGS, latent_dim=LATENT_DIM,
              num_channels=NUM_CHANNELS)
vqvae.load_weights(VQVAE_WEIGHTS_PATH + 'trained_model_weights')


# Generate and plot some VQ-VAE reconstructions from test set
def generate_vqvae_images(images):
    """
    Plot + saves input, codebook
    :param images: Images to pass throug VQ-VAE
    :return: A bunch of stuff
    """
    # Reconstruction
    reconstructed = vqvae.predict(images)

    # Code (flattened)
    encoder_outputs = vqvae.get_encoder().predict(images)
    encoder_outputs_flat = encoder_outputs.reshape(-1, encoder_outputs.shape[-1])
    codebook_indices = vqvae.get_vq().get_codebook_indices(encoder_outputs_flat)

    # Reshaped
    codebook_indices = codebook_indices.numpy().reshape(encoder_outputs.shape[:-1])

    # Unshift + reshape reconstructions and original images
    plt.figure()
    for i in range(len(images)):
        test_image = tf.reshape(images[i], (1, IMG_SHAPE, IMG_SHAPE, NUM_CHANNELS)) + PIXEL_SHIFT
        reconstructed_image = tf.reshape(reconstructed[i], (1, IMG_SHAPE, IMG_SHAPE, NUM_CHANNELS)) + PIXEL_SHIFT

        codebook_image = codebook_indices[i]

        plt.subplot(1, 3, 1)
        plt.imshow(tf.squeeze(test_image))
        plt.title("Sample Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(codebook_image)
        plt.title("Codebook deconstruction")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(reconstructed_image))
        plt.title("VQVAE Reconstruction")
        plt.axis("off")

        plt.savefig(RESULTS_PATH + f'vq_vae_reconstructions_{i}.png')

        ssim = tf.math.reduce_sum(tf.image.ssim(test_image, reconstructed_image,
                                                max_val=1.0)).numpy()

        main_stdout = sys.stdout
        with open(RESULTS_PATH + 'main_results.txt', 'a') as f:
            sys.stdout = f
            print(f"Test Image {i} and Reconstruction SSIM: ", ssim)
            sys.stdout = main_stdout


for sample_batch in test_data.take(1).as_numpy_iterator():
    sample_batch = sample_batch[:NUM_IMAGES_TO_SHOW]
    generate_vqvae_images(sample_batch)

# Load trained model
pixelCNN = PixelCNN(num_res=NUM_RESIDUAL_LAYERS, num_pixel_B=NUM_PIXEL_B_LAYERS,
                    num_encoded=NUM_EMBEDDINGS, num_filters=NUM_PIXEL_FILTERS,
                    kernel_size=PIXEL_KERNEL_SIZE, activation=PIXEL_ACTIVATION)
pixelCNN.load_weights(PIXEL_WEIGHTS_PATH + "trained_model_weights")


"""
Generation via Codebook Sampling for novel Image generation
ref: https://keras.io/examples/generative/vq_vae/
"""

# Create a mini sampler model
inputs = layers.Input(shape=pixelCNN.input_shape[1:])
outputs = pixelCNN(inputs)
categorical_layer = tfp.layers.DistributionLambda(tfp.distributions.Categorical)
outputs = categorical_layer(outputs)
sampler = keras.Model(inputs, outputs)

# Create empty array of priors
batch = 5
priors = np.zeros(shape=(batch,) + (pixelCNN.input_shape)[1:])
batch, rows, cols = priors.shape

for row in range(rows):
    for col in range(cols):
        probs = sampler.predict(priors)
        priors[:, row, col] = probs[:, row, col]

print(f'Prior shape: {priors.shape}')

# Perform an embedding lookup
pretrained_embeddings = vqvae.get_vq().get_codebook_indices()
priors_one = tf.one_hot(priors.astype('int32'), NUM_EMBEDDINGS).numpy()
qtised = tf.matmul(
    priors_one.astype('float32', pretrained_embeddings, transpose_b=True)
)
qtised = tf.reshape(qtised, (-1, *(IMG_SHAPE[1:])))

# Generate novel Images
decoder = vqvae.get_layer('decoder')
generated_samples = decoder.predict(qtised)

for i in range(batch):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i])
    plt.title('Code')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title('Generated Sample')
    plt.axis('off')
    plt.show()


