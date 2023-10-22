import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from tensorflow import keras
import os

from utils import dice_coefficient


# Test and visualize model predictions with a set amount of test inputs.
def test_visualise_model_predictions(model, test_gen, output_dir, timestr, number_of_predictions):
    test_range = np.arange(0, stop=number_of_predictions, step=1)
    
    for i in test_range: 
        current = next(islice(test_gen, i, None))
        image_input = current[0]  # Image tensor
        mask_truth = current[1]  # Mask tensor
        # debug statements to check the types of the tensors and find the division by zero
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
