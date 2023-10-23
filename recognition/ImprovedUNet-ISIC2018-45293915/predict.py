import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import math
import time
from itertools import islice

from utils import dice_coefficient
from dataset import DataLoader

BATCH_SIZE = 2  # set the batch_size
STEPS_PER_EPOCH_TEST = math.floor(1000 / BATCH_SIZE)


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
def test_and_visualise_predictions(model, test_data, output_dir, timestr, number_of_predictions):

    test_loss, test_accuracy, test_dice = \
        model.evaluate(test_data, steps=STEPS_PER_EPOCH_TEST, verbose=2, use_multiprocessing=False)
    print("Test Accuracy: " + str(test_accuracy))
    print("Test Loss: " + str(test_loss))
    print("Test DSC: " + str(test_dice) + "\n")

    test_range = np.arange(0, stop=number_of_predictions, step=1)
    
    for i in test_range: 
        current = next(islice(test_data, i, None))
        image_input = current[0]  # Image tensor
        mask_truth = current[1]  # Mask tensor
        # debug statements to check the types of the tensors and find the division by zero
        test_pred = model.predict(image_input, steps=1, use_multiprocessing=False)[0]
        truth = mask_truth[0]
        original = image_input[0]
        probabilities = keras.preprocessing.image.img_to_array(test_pred)
        test_dice = dice_coefficient(truth, test_pred, axis=None)


        # Create a unique filename for each visualization
        filename = os.path.join(output_dir, f"test_visualisation_{i}_{timestr}.png")

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

    print("Visualisations of test output saved in the 'output' folder.")

if __name__ == "__main__":
    # Constants related to preprocessing
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

    print("\nPREPROCESSING IMAGES")
    # print number of images in each directory
    test_data = DataLoader(
        test_dir, test_groundtruth_dir, image_mode, mask_mode, image_height, image_width, batch_size, seed,
        shear_range, zoom_range, horizontal_flip, vertical_flip, fill_mode)
    test_data = test_data.create_data_generators()


    model_name = "WORK.keras"
    output_dir = "output"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print("\nTESTING MODEL")
    test_and_visualise_predictions(model_name, test_data, output_dir, timestr, number_of_predictions)