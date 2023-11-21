"""
predict.py

Author: Ethan Jones
Student ID: 44829531
COMP3710 OASIS brain StyleGAN project
Semester 2, 2023
"""

import os
import tensorflow as tf
from tensorflow import keras
from train import StyleGAN

# Input paths
INPUT_IMAGES_PATH = "/home/groups/comp3710/OASIS/keras_png_slices_train"
INPUT_GENERATOR_WEIGHTS_PATH = ""
INPUT_DISCRIMINATOR_WEIGHTS_PATH = ""

# Output paths
RESULT_IMAGE_PATH = "figures"
RESULT_WEIGHT_PATH = "weights"

# Output parameters
IMG_COUNT = 3
PLOT_LOSS = True

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 32


def main():
    """
    Main function to execute the StyleGAN workflow.
    """
    directory_name = os.path.dirname(__file__)

    # Create directories for output files
    if RESULT_IMAGE_PATH != "":
        filepath = os.path.join(directory_name, RESULT_IMAGE_PATH)
        if not os.path.exists(filepath):
            os.mkdir(filepath)

    if RESULT_WEIGHT_PATH != "":
        filepath = os.path.join(directory_name, RESULT_WEIGHT_PATH)
        if not os.path.exists(filepath):
            os.mkdir(filepath)

    # Initialise the StyleGAN model
    style_gan = StyleGAN(epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Load the pre-trained model when weights are provided
    if INPUT_GENERATOR_WEIGHTS_PATH != "" and \
            INPUT_DISCRIMINATOR_WEIGHTS_PATH != "":
        style_gan.built = True

        generator_filepath = os.path.join(directory_name,
                                          INPUT_GENERATOR_WEIGHTS_PATH)
        discriminator_filepath = os.path.join(directory_name,
                                              INPUT_DISCRIMINATOR_WEIGHTS_PATH)

        style_gan.generator.load_weights(generator_filepath)
        style_gan.discriminator.load_weights(discriminator_filepath)
    # Train the model
    else:
        style_gan.train(dataset_path=INPUT_IMAGES_PATH,
                        result_image_path=RESULT_IMAGE_PATH,
                        image_count=IMG_COUNT,
                        result_weight_path=RESULT_WEIGHT_PATH,
                        plot_loss=PLOT_LOSS)

    # Generate an image and display
    output(style_gan)


def output(style_gan):
    """
    Generates and displays an image using the provided StyleGAN model.
    """
    generator_inputs = style_gan.generator_inputs()

    # Generate an image using the trained model
    output_images = style_gan.generator(generator_inputs)
    output_images *= 256
    output_images.numpy()

    # Convert the tensor to an image and display
    img = keras.preprocessing.image.array_to_img(output_images[0])
    img.show(title="Generated Image")


# Run main() if this file is run directly
if __name__ == "__main__":
    main()
