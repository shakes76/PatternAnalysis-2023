import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

from utils import dice_coefficient


def save_prediction(original, probabilities, truth, dice_coeff, filename):
    # Create a subplot for the visualization
    figure, axes = plt.subplots(1, 3)
    
    # Plot and save the input image
    axes[0].title.set_text('Input')
    axes[0].imshow(original, vmin=0.0, vmax=1.0)
    axes[0].set_axis_off()
    
    # Plot and save the model's output
    axes[1].title.set_text('Dice Coeff: ' + str(dice_coeff.numpy()))
    axes[1].imshow(probabilities, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_axis_off()
    
    # Plot and save the ground truth
    axes[2].title.set_text('Ground Truth')
    axes[2].imshow(truth, cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_axis_off()

    # Save the visualisation to the output folder
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# Evaluate and visualise model predictions on the validation set.
def validate_and_visualise_predictions(model, validation_data, output_dir, timestr, number_of_predictions):
    # Initialise variables to calculate statistics
    total_dice_coefficient = 0.0
    total_samples = 0

    for i, (image_input, mask_truth) in enumerate(validation_data):
        # Predict using the model
        predictions = model.predict(image_input, steps=1, use_multiprocessing=False)
        prediction = predictions[0]
        truth = mask_truth[0]
        original = image_input[0]
        probabilities = keras.preprocessing.image.img_to_array(prediction)
        dice_coeff = dice_coefficient(truth, prediction, axis=None)

        # Create a unique filename for each visualisation
        filename = os.path.join(output_dir, f"visualisation_{i}_{timestr}.png")

        # Save the visualisation
        if i < number_of_predictions:
            save_prediction(original, probabilities, truth, dice_coeff, filename)

        # Accumulate statistics
        total_dice_coefficient += dice_coeff
        total_samples += 1


    # Calculate and print average Dice coefficient for the entire validation set
    average_dice_coefficient = total_dice_coefficient / total_samples
    print("Average Dice Coefficient on Validation Set:", average_dice_coefficient)

    print("Visualisations saved in the 'output' folder.")
