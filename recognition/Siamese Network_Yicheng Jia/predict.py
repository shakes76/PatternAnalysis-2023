"""
    File name: predict.py
    Author: Yicheng Jia
    Date created: 27/09/2023
    Date last modified: 26/10/2023
    Python Version: 3.11.04
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from modules import SiameseNetwork


def check_CUDA():
    # Check if CUDA is available and set the device accordingly
    current_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return current_device


def read_from_checkpoint(current_model_pth: str):
    # read the model
    current_checkpoint = torch.load(current_model_pth)
    net.load_state_dict(current_checkpoint['module_state_dict'])
    current_training_epoch = current_checkpoint['epoch']

    return current_checkpoint, current_training_epoch


def predict_progress(predicted_module, current_device, predicted_loop_dataloader):
    # Predict Loop
    predicted_module.eval()

    # Initial values
    predicted_total_losses = 0.0
    predicted_correct_numbers = 0.0
    predicted_total_labels = 0.0
    print("Start predicting!")

    # Loss function
    criterion = nn.BCELoss()

    # Progress bar
    with torch.no_grad():
        for index, image_data in tqdm(enumerate(predicted_loop_dataloader), total=len(predicted_loop_dataloader)):
            img_0, img_1, image_label = image_data  # Get the images and labels from the data
            img_0, img_1, image_label = img_0.to(current_device), img_1.to(current_device), image_label.to(current_device)  # Move the images and labels to the appropriate device

            predicted_total_outputs = predicted_module(img_0, img_1).squeeze()  # Adjust the size of the output to match the target
            predicted_loss = criterion(predicted_total_outputs, image_label)  # Get the loss

            # Calculate the accuracy
            predicted_label = torch.where(predicted_total_outputs > 0.5, torch.tensor(1, device=current_device), torch.tensor(0, device=current_device))
            predicted_total_labels += len(image_label)
            predicted_correct_numbers += predicted_label.eq(image_label.view_as(predicted_label)).sum().item()
            predicted_total_losses += predicted_loss.item()

    print("Testing finished, printing results...")

    predict_average_loss = predicted_total_losses / len(predicted_loop_dataloader)
    accuracy = 100. * predicted_correct_numbers / predicted_total_labels

    print(f"Test_Average_Loss:{predict_average_loss:.4f},Test_Accuracy:{accuracy:.4f}%")


if __name__ == '__main__':

    device = check_CUDA()
    net = SiameseNetwork().to(device)

    ROOT_DIR = "AD_NC"
    # Create data loaders for training and testing datasets
    current_train_dataset, test_dataset, _ = dataset.get_datasets(ROOT_DIR)
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=256)

    # read the model from the checkpoint of the previous module
    checkpoint, current_epoch = read_from_checkpoint('module_best_test_accuracy.pth')

    predict_progress(net, device, test_dataloader)
