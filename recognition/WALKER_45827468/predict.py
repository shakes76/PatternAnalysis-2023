from dataset import ISICDataset, calc_mean_std
from modules import ImprovedUNet, DiceLoss

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

BATCH_SIZE = 10

# device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if not torch.backends.mps.is_available():
    print("Warning MPS not found. Using CPU")

print('beep boop')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.7079, 0.5915, 0.5469], [0.1543, 0.1629, 0.1780]),
    transforms.Resize((256, 256))
])

TEST_DATA_PATH = "./ISIC-2017_Validation_Data"
TEST_MASK_PATH = "./ISIC-2017_Validation_Part1_GroundTruth"

test = ISICDataset(TEST_DATA_PATH, TEST_MASK_PATH, transform=transform)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

# load in trained model
ImpUNET = torch.load("impUNetMODEL.pth")
ImpUNET.to(device)

# run test set through model, get predictions
images = []
masks = []
prediction = []
ImpUNET.eval()
with torch.no_grad():
    for im,mask in test_loader:
        im = im.to(device)
        mask = mask.to(device)
        pred = ImpUNET(im)
        prediction.append(pred.cpu())
        masks.append(mask.cpu())
        images.append(im.cpu())
        
        
# find dice coefficient

# plot example
fig,ax = plt.subplots(1, 3, figsize=(10,10))
ax[0].imshow(images[0])
ax[1].imshow(masks[0])
ax[2].imshow(pred[0])
fig.show()