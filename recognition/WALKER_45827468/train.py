from dataset import ISICDataset, calc_mean_std
from modules import ImprovedUNet

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 1
NUM_EPOCH = 10
LEARNING_RATE = 1e-4

# device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if not torch.backends.mps.is_available():
    print("Warning MPS not found. Using CPU")

print('beep boop')

TRAIN_DATA_PATH = "./ISIC-2017_Training_Data"
TRAIN_MASK_PATH = "./ISIC-2017_Training_Part1_GroundTruth"

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize()
# ])

train = ISICDataset(TRAIN_DATA_PATH, TRAIN_MASK_PATH, transform=transforms.ToTensor())
train_loader = DataLoader(train, batch_size=BATCH_SIZE)

VALID_DATA_PATH = "./ISIC-2017_Validation_Data"
VALID_MASK_PATH = "./ISIC-2017_Validation_Part1_GroundTruth"

# valid = ISICDataset(TRAIN_DATA_PATH, TRAIN_MASK_PATH, transform=transforms.ToTensor())
# valid_loader = DataLoader(valid, batch_size=BATCH_SIZE)

# print(calc_mean_std(train_loader))

ImpUNET = ImprovedUNet()
ImpUNET.to(device)

lossFunc = nn.BCELoss()
opt = torch.optim.Adam(ImpUNET.parameters(), lr=LEARNING_RATE)

ImpUNET.train()

# losses
loss_train = []
loss_valid = []

for epoch in range(NUM_EPOCH):

    # iterate through training set
    for im,mask in train_loader:
        
        im = im.to(device)
        mask = mask.to(device)
        
        pred = ImpUNET(im)
        loss = lossFunc(pred, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(ImpUNET, "impUNetMODEL.pth")