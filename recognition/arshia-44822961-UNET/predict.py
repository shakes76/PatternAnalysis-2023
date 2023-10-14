import torch
import numpy as np

from dataset import ISICDataset
from train import load_data, data_transform, BATCH_SIZE

# model file 
MODEL_FILE_PATH = "improved_UNET.pth"

# plot output paths 
OUTPUT_DIR_PATH = "~/report1"

# data 
TEST_DATA_PATH = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input"
TEST_MASK_PATH = "~/report1/ISIC2018_Task1_Test_GroundTruth"

def perform_predictions(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())
    return predictions

def dice_coefficient(predicted, target):
    smooth = 1.0  # Add a smoothing term to prevent division by zero
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


if __name__ == "__main__":
    # connect to gpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in data and model
    test_loader = load_data(TEST_DATA_PATH, TEST_MASK_PATH, data_transform, batch_size=BATCH_SIZE)
    model = torch.load(MODEL_FILE_PATH)

    # perform predictions 
    predictions = perform_predictions(model, test_loader, device)

    for i, (pred, (data, target)) in enumerate(zip(predictions, test_loader)):
        dice = dice_coefficient(pred, target)
        print(f'Dice Coefficient for image {i}: {dice}')



