from dataset import ISICDataset, calc_mean_std
from modules import ImprovedUNet, DiceLoss

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 10
NUM_EPOCH = 10
LEARNING_RATE = 1e-4

# device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if not torch.backends.mps.is_available():
    print("Warning MPS not found. Using CPU")

print('beep boop')

TRAIN_DATA_PATH = "./ISIC-2017_Training_Data"
TRAIN_MASK_PATH = "./ISIC-2017_Training_Part1_GroundTruth"

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.7079, 0.5915, 0.5469], [0.1543, 0.1629, 0.1780]),
    transforms.Resize((256, 256))
])

train = ISICDataset(TRAIN_DATA_PATH, TRAIN_MASK_PATH, transform=transform)
train_loader = DataLoader(train, batch_size=BATCH_SIZE)

VALID_DATA_PATH = "./ISIC-2017_Validation_Data"
VALID_MASK_PATH = "./ISIC-2017_Validation_Part1_GroundTruth"

valid = ISICDataset(TRAIN_DATA_PATH, TRAIN_MASK_PATH, transform=transform)
valid_loader = DataLoader(valid, batch_size=BATCH_SIZE)

# mean,std = calc_mean_std(train_loader)
# print(mean,std)

ImpUNET = ImprovedUNet()
ImpUNET.to(device)

lossFunc = nn.BCELoss()
opt = torch.optim.Adam(ImpUNET.parameters(), lr=LEARNING_RATE)

ImpUNET.train()

# losses
loss_train = []
loss_valid = []

for epoch in range(NUM_EPOCH):

    ImpUNET.train()
    total_loss = 0
    # iterate through training set
    for im,mask in train_loader:
        
        im = im.to(device)
        mask = mask.to(device)
        
        opt.zero_grad()
        pred = ImpUNET(im)
        loss = lossFunc(pred, mask)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print("EPOCH", epoch, ":")
    print("     TRAINING LOSS:", total_loss / float(len(train_loader)))
    loss_train.append(total_loss / float(len(train_loader)))
    
    ImpUNET.eval()
    total_loss = 0
    with torch.no_grad():
        for im,mask in valid_loader:
            im = im.to(device)
            mask = mask.to(device)
            
            pred = ImpUNET(im)
            loss = lossFunc(pred, mask)
            
            total_loss += loss.item()
    print("     VALIDATION LOSS:", total_loss / float(len(valid_loader)))
    loss_valid.append(total_loss / float(len(valid_loader)))
    
# save model and associated losses
torch.save(ImpUNET, "impUNetMODELsig.pth")
import pandas as pd
losses = pd.DataFrame()
losses['TRAIN'] = loss_train
losses['VALID'] = loss_valid
losses.to_csv('lossessig.csv')

# plot losses
import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_train, label="training loss")
plt.plot(loss_valid, label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE)")
plt.legend()
plt.savefig("lossgraph.jpg")