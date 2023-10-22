import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import math
from keras.callbacks import Callback
import time

import modules as layers
from dataset import pre_process_data

# string modifier for saving output files based on time
timestr = time.strftime("%Y%m%d-%H%M%S")
output_dir = "output"

# Constants related to training
EPOCHS = 5
LEARNING_RATE = 0.0005
BATCH_SIZE = 2  # set the batch_size
IMAGE_HEIGHT = 512  # the height input images are scaled to
IMAGE_WIDTH = 512  # the width input images are scaled to
CHANNELS = 3
STEPS_PER_EPOCH_TRAIN = math.floor(2076 / BATCH_SIZE)
STEPS_PER_EPOCH_TEST = math.floor(519 / BATCH_SIZE)
NUMBER_SHOW_TEST_PREDICTIONS = 3

# Define a callback to calculate Dice coefficient after each epoch
class DiceCoefficientCallback(Callback):
    def __init__(self, test_gen, steps_per_epoch_test):
        self.test_gen = test_gen
        self.steps_per_epoch_test = steps_per_epoch_test
        self.dice_coefficients = []

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy, test_dice = \
            self.model.evaluate(self.test_gen, steps=self.steps_per_epoch_test, verbose=0, use_multiprocessing=False)
        self.dice_coefficients.append(test_dice)
        print(f"Epoch {epoch + 1} - Test Dice Coefficient: {test_dice:.4f}")


# Plot the accuracy and loss curves of model training.
def plot_accuracy_loss(track):
    plt.figure(0)
    plt.plot(track.history['accuracy'])
    plt.plot(track.history['loss'])
    plt.title('Loss & Accuracy Curves')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    
    # Generate a unique filename based on the current date and time
    now = datetime.datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"accuracy_loss_plot_{timestr}.png")
    
    # Save the plot to the output folder
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Accuracy and loss plot saved as '{filename}'.")


# Metric for how similar two sets (prediction vs truth) are.
# Implementation based off https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# DSC = (2|X & Y|) / (|X| + |Y|) -> 'soft' dice coefficient.
def dice_coefficient(truth, pred, eps=1e-7, axis=(1, 2, 3)):
    numerator = (2.0 * (tf.reduce_sum(pred * truth, axis=axis))) + eps
    denominator = tf.reduce_sum(pred, axis=axis) + tf.reduce_sum(truth, axis=axis) + eps
    dice = tf.reduce_mean(numerator / denominator)
    return dice


# Loss function - DSC distance.
def dice_loss(truth, pred):
    return 1.0 - dice_coefficient(truth, pred)


# Compile and train the model, evaluate test loss and accuracy.
def train_model_check_accuracy(train_gen, test_gen):
    model = layers.improved_unet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=dice_loss, metrics=['accuracy', dice_coefficient])
    track = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        use_multiprocessing=False)
    plot_accuracy_loss(track)

    print("\nEvaluating test images...")
    test_loss, test_accuracy, test_dice = \
        model.evaluate(test_gen, steps=STEPS_PER_EPOCH_TEST, verbose=2, use_multiprocessing=False)
    print("Test Accuracy: " + str(test_accuracy))
    print("Test Loss: " + str(test_loss))
    print("Test DSC: " + str(test_dice) + "\n")
    return model

# Test and visualize model predictions with a set amount of test inputs.
def test_visualise_model_predictions(model, test_gen):
    test_range = np.arange(0, stop=NUMBER_SHOW_TEST_PREDICTIONS, step=1)
    
    for i in test_range:
        current = next(islice(test_gen, i, None))
        image_input = current[0]  # Image tensor
        mask_truth = current[1]  # Mask tensor
        test_pred = model.predict(image_input, steps=1, use_multiprocessing=False)[0]
        truth = mask_truth[0]
        original = image_input[0]
        probabilities = keras.preprocessing.image.img_to_array(test_pred)
        test_dice = dice_coefficient(truth, test_pred, axis=None)

        # Create a unique filename for each visualization
        filename = os.path.join(output_dir, f"visualization_{i}_{timestr}.png")

        # Create a subplot for the visualization
        figure, axes = plt.subplots(1, 3)
        
        # Plot and save the input image
        axes[0].title.set_text('Input')
        axes[0].imshow(original, vmin=0.0, vmax=1.0)
        axes[0].set_axis_off()
        
        # Plot and save the model's output
        axes[1].title.set_text('Output (DSC: ' + str(test_dice.numpy()) + ")")
        axes[1].imshow(probabilities, cmap='gray', vmin=0.0, vmax=1.0)
        axes[1].set_axis_off()
        
        # Plot and save the ground truth
        axes[2].title.set_text('Ground Truth')
        axes[2].imshow(truth, cmap='gray', vmin=0.0, vmax=1.0)
        axes[2].set_axis_off()

        # Save the visualization to the output folder
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    print("Visualizations saved in the 'output' folder.")


# Run the test driver.
def main():
    print("\nPREPROCESSING IMAGES")
    train_gen, test_gen = pre_process_data()
    print("\nTRAINING MODEL")
    model = train_model_check_accuracy(train_gen, test_gen)
    # Save the trained model to a file
    print("\nSAVING MODEL")
    model.save("my_model.keras")
    print("\nVISUALISING PREDICTIONS")
    test_visualise_model_predictions(model, test_gen)

    # Initialize the callback for Dice coefficient
    dice_coefficient_callback = DiceCoefficientCallback(test_gen, STEPS_PER_EPOCH_TEST)
    
    print("\nPLOTTING DICE COEFFICIENT")
    plt.figure()
    plt.plot(dice_coefficient_callback.dice_coefficients)
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.show()

    print("COMPLETED")
    return 0


if __name__ == "__main__":
    main()