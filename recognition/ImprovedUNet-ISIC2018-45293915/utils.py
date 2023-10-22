from keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import os

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
def plot_accuracy_loss(track, output_dir, timestr):
    plt.figure(0)
    plt.plot(track.history['accuracy'])
    plt.plot(track.history['loss'])
    plt.title('Loss & Accuracy Curves')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'])
    
    # Generate a unique filename based on the current date and tim
    filename = os.path.join(output_dir, f"accuracy_loss_plot_{timestr}.png")
    
    # Save the plot to the output folder
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Accuracy and loss plot saved as '{filename}'.")


# Plot the dice coefficient curve and save it as an image
def save_dice_coefficient_plot(dice_history, output_dir, timestr):
    filename = os.path.join(output_dir, f"dice_coefficient_plot_{timestr}.png")
    plt.figure(1)
    plt.plot(dice_history)
    plt.title('Dice Coefficient Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.savefig(filename)  # Save the plot as an image
    plt.close()  # Close the figure to release resources
    print("Dice coefficeint saved as " + filename + ".")