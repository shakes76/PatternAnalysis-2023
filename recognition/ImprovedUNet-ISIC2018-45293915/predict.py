import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

from utils import dice_coefficient
from visualise import save_prediction
from dataset import DataLoader

# Evaluate and visualise model predictions on the validation set.
def validate_and_visualise_predictions(model, test_data, output_dir, timestr, number_of_predictions):
    # Initialise variables to calculate statistics
    total_dice_coefficient = 0.0
    total_samples = 0

    for i, (image_input, mask_truth) in enumerate(test_data):
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

if __name__ == "__main__":
    test_dir = "datasets/test_input"
    test_groundtruth_dir = "datasets/test_groundtruth"
    image_mode = "rgb"
    mask_mode = "grayscale"
    image_height = 512
    image_width = 512
    batch_size = 2
    seed = 45
    shear_range = 0.1
    zoom_range = 0.1
    horizontal_flip = True
    vertical_flip = True
    fill_mode = 'nearest'
    number_of_predictions = 3
    test_data = DataLoader(
        test_dir, test_groundtruth_dir, image_mode, mask_mode, image_height, image_width, batch_size, seed,
        shear_range, zoom_range, horizontal_flip, vertical_flip, fill_mode)
    test_data = test_data.create_data_generators()