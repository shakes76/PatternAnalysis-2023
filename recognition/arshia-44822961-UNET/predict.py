import torch
import numpy as np
import matplotlib.pyplot as plt
import os 

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
    inputs = []
    targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())
    return predictions, inputs, targets

def dice_coefficient(predicted, target):
    smooth = 1.0  # Add a smoothing term to prevent division by zero
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


# histogram of coefficents 
def plot_histogram(dice_coeffs, directory):
    plt.hist(dice_coeffs, bins=10, edgecolor='black')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Frequency')
    plt.title('Histogram of Dice Coefficients')
    plt.savefig(os.path.join(directory, 'dice_dist.png'))
    plt.close()

# images of thre min, max and avg dice coefficients. 
def visualize_extreme_cases(predictions, inputs, targets, directory):
    coefficients = [dice_coefficient(pred, target) for pred, target in zip(predictions, targets)]
    
    min_idx = np.argmin(coefficients)
    max_idx = np.argmax(coefficients)
    avg_idx = (np.abs(coefficients - np.mean(coefficients))).argmin()
    
    for idx, name in zip([min_idx, max_idx, avg_idx], ['min', 'max', 'avg']):
        img, gt, pred = inputs[idx], targets[idx], predictions[idx]
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img.permute(1, 2, 0)) 
        axs[0].set_title('Original Image')
        axs[1].imshow(gt.squeeze(), cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[2].imshow(pred.squeeze(), cmap='gray')
        axs[2].set_title(f'Predicted Mask\nDice Coefficient: {coefficients[idx]:.2f}')
        
        for ax in axs:
            ax.axis('off')
        
        plt.savefig(f'{directory}/{name}_dice_coefficient.png', bbox_inches='tight')
        plt.close(fig)



if __name__ == "__main__":
    # connect to gpu 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in data and model
    test_loader = load_data(TEST_DATA_PATH, TEST_MASK_PATH, data_transform, batch_size=BATCH_SIZE)
    model = torch.load(MODEL_FILE_PATH)

    # perform predictions 
    predictions, inputs, targets = perform_predictions(model, test_loader, device)

    for i, (pred, data, target) in enumerate(zip(predictions, input_images, target_images)):
        dice = dice_coefficient(pred, target)
        print(f'Dice Coefficient for image {i}: {dice}')

    # plot dice coefficients in a histogram. 
    plot_histogram(dice, OUTPUT_DIR_PATH)

    # plot three examples of images and their respective segments. 
    visualize_extreme_cases(predictions, inputs, targets, OUTPUT_DIR_PATH)



