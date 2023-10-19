"""
Loads the model and runs it on the test set. Saves the predictions to a folder for visualization.
and calculates the dice score on the test set.
"""

from dataset import ISICDataset
import torch
from utils import *
from modules import ImprovedUNet
from dataset import ISICDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from global_params import *

# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1 # We want to test one image at a time
NUM_WORKERS = 1
CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'

def main():
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device=device)
    load_checkpoint(torch.load(CHECKPOINT_DIR), model)

    # Defining the test loader
    test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
    test_set = ISICDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # calculate the dice score on the test set
    dice_score = calc_dice_score(model, test_loader, device=device, verbose=True)
    print(f'Test Dice Score: {dice_score:.4f}')
    
    save_predictions_as_imgs(test_loader, model, num=30, folder='saved_images/', device=device)

    plot_samples(6, title='Predictions', include_image=True)

if __name__ == '__main__':
    main()