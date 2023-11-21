"""
Example usage of the trained model on images in the test set.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import *
from config import *
from modules import SuperResolutionModel
from train import train_model
from generate import generate_model_output


def main():
    model = SuperResolutionModel(upscale_factor=dimension_reduce_factor)
    # Try to load model from file
    try:
        model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))
    except IOError as err:
        print("Couldn't load model from file:", model_filename)
        # Prompt user for training
        redo = input("Would you like to train the model? (y/n):")
        while redo not in ['y', 'n', 'Y', 'N']:
            redo = input("Would you like to train the model? (y/n):")
        if redo in ['y', 'Y']:
            # Train model from scratch
            train_model()
            # Restart predict
            main()
        else:
            print("Exiting...")
            exit(1)

    print("Loaded model from file:", model_filename)

    # Display the outputs from some testing images
    data_loader = get_test_dataloader()
    print("Displaying model output from test images...")
    num_show = 5
    for i in range(num_show):
        generate_model_output(model, data_loader, show=True, plot_title=f"Test Image {i+1}/{num_show}")


if __name__ == '__main__':
    main()