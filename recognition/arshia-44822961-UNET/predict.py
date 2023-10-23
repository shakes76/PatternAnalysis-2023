"""
File: predict.py
Author: Arshia Sharma

Description: Performs inference using trained model
Dependencies: torch torchvision numpy matplotlib 
"""
# libraries 
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from torch.utils.data import DataLoader

# import from local files  
from train import load_data, data_transform, dice_coefficient
from dataset import ISICDataset

# UPDATE IF NOT SAVED IN BASE DIRECTORY. 
MODEL_FILE_PATH = "improved_UNET.pth"

# UPDATE WITH PATH TO YOUR TEST DATA
TEST_DATA_PATH = "ISIC2018/ISIC2018_Task1-2_Validation_Input"
TEST_MASK_PATH = "ISIC2018/ISIC2018_Task1_Validation_GroundTruth"

"""
    Tests improved unet on trained model. 
    Calcualtes dice coeficient for each image and corresponding ground truth. 

    Parameters:
    - model (nn.Module): The trained model to be tested.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (str): The device (e.g., 'cuda' or 'cpu') to run the evaluation on.

    Returns:
    - dice_scores (list): List of Dice coefficients for each image in the test dataset.
"""
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    
    dice_scores = [] # stores dice scores. 

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            dice = dice_coefficient(outputs, targets)
            dice_scores.append(dice.item())

    return dice_scores

"""
Load in test data and return dataset loader object. 

"""
def load_data(img_path, labels_path, transform, batch_size, shuffle=True):
    test_dataset = ISICDataset(img_path, labels_path, transform) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle) 
    return test_loader

"""
    Visualises model image, predictions and ground truth on first three images from test loader.

    Parameters:
    - model (nn.Module): The trained model used for making predictions.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - device (str): The device (e.g., 'cuda' or 'cpu') to run the visualization on.
    - num_images (int): The number of images to visualize (default is 3).

    """
def visualise_predictions(model, test_loader, device, num_images=3):
    model.eval()  # Set the model to evaluation mode

    image_count = 0  # Keep track of the number of images processed

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            # get prediction 
            outputs = model(inputs)

            # Convert PyTorch tensors to NumPy arrays
            input_image = inputs[0].cpu().numpy()  
            target_image = targets[0].cpu().numpy()
            predicted_image = outputs[0].cpu().numpy()

            # Create a side-by-side visualization for three images, prediction, ground truth. 
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(input_image[0], cmap='gray')  

            plt.subplot(1, 3, 2)
            plt.title("Model Prediction")
            plt.imshow(predicted_image[0], cmap='gray')  

            plt.subplot(1, 3, 3)
            plt.title("Ground Truth")
            plt.imshow(target_image[0], cmap='gray')  

            plt.show()

            image_count += 1

            if image_count >= num_images:
                break

"""
    Plots dice coefficients of the whole test dataset.
    Takes an array of dice scores as input. 
"""
def plot_dice(dice):
    x_values = np.arange(len(dice))  # Generate x-values as indices
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, dice, marker='o', linestyle='-')
    plt.xlabel("Image Index")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Coefficient across test inputs")
    plt.grid(True)
    plt.show()


"""
    Driver method 

"""
if __name__ == "__main__":
    # connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in data and model
    test_loader = load_data(TEST_DATA_PATH, TEST_MASK_PATH, data_transform, batch_size=1)
    model = torch.load(MODEL_FILE_PATH)

    # perform predictions
    dice_scores = test(model, test_loader, device)
    average_dice = np.mean(dice_scores)
    print(f"Average Dice Coefficient: {average_dice:.4f}")

    # plot dice scores across the dataset.
    plot_dice(dice_scores)

    # plot three examples of images, prediction and truth. 
    visualise_predictions(model, test_loader,device)


