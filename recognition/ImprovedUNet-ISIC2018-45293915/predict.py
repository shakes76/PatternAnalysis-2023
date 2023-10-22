import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

from dataset import pre_process_data  # Import your data preprocessing function here


NUMBER_SHOW_TEST_PREDICTIONS = 3 # Number of test predictions to show


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


# Test and visualize model predictions with a set amount of test inputs.
def test_visualise_model_predictions(model, test_gen):
    print(model)
    print(test_gen)
    test_range = np.arange(0, stop=NUMBER_SHOW_TEST_PREDICTIONS, step=1)
    figure, axes = plt.subplots(NUMBER_SHOW_TEST_PREDICTIONS, 3)
    for i in test_range:
        current = next(islice(test_gen, i, None))
        image_input = current[0]  # Image tensor
        mask_truth = current[1]  # Mask tensor
        test_pred = model.predict(image_input, steps=1, use_multiprocessing=False)[0]
        truth = mask_truth[0]
        original = image_input[0]
        probabilities = keras.preprocessing.image.img_to_array(test_pred)
        test_dice = dice_coefficient(truth, test_pred, axis=None)

        axes[i][0].title.set_text('Input')
        axes[i][0].imshow(original, vmin=0.0, vmax=1.0)
        axes[i][0].set_axis_off()
        axes[i][1].title.set_text('Output (DSC: ' + str(test_dice.numpy()) + ")")
        axes[i][1].imshow(probabilities, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][1].set_axis_off()
        axes[i][2].title.set_text('Ground Truth')
        axes[i][2].imshow(truth, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][2].set_axis_off()
    plt.axis('off')
    plt.show()

def main():
    print("\nPREPROCESSING IMAGES")
    train_gen, test_gen = pre_process_data()
    print("\nLOADING TRAINED MODEL")
    model = keras.models.load_model("path_to_trained_model.h5")  # Load your trained model here
    print("\nVISUALISING PREDICTIONS")
    test_visualise_model_predictions(model, test_gen)

if __name__ == "__main__":
    main()
