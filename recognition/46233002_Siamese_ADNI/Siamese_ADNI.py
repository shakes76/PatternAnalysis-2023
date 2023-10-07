import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
import random
from datasets import ADDataTrain, ADDataTest

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

# Path
path = '/home/Student/s4623300'

# Hyper-parameters
#num_epochs = 10
#learning_rate = 0.001
batch_size = 8

# Get file names
train_ads = os.listdir(path + "/data/AD_NC/train/AD") # len = 10400
train_ncs = os.listdir(path + "/data/AD_NC/train/NC") # len = 11120
test_ads = os.listdir(path + "/data/AD_NC/test/AD")   # len = 4460
test_ncs = os.listdir(path + "/data/AD_NC/test/NC")   # len = 4540

# Balanced test set 
if (len(test_ads) < len(test_ncs)):
    test_ncs = random.sample(test_ncs, len(test_ads))
else:
    test_ads = random.sample(test_ads, len(test_ncs))

# Extract validation set from test set 
val_size = 0.5
val_ads = [x.pop(0) for x in random.sample([test_ads]*len(test_ads), int(len(test_ads)*val_size))]
val_ncs = [x.pop(0) for x in random.sample([test_ncs]*len(test_ncs), int(len(test_ncs)*val_size))]

# Check data distribution
print("Distribution of ads/ncs")
print("Train set: ", len(train_ads), len(train_ncs))
print("Test set: ", len(test_ads), len(test_ncs))
print("Validation set: ", len(val_ads), len(val_ncs))
print()

# Transformer
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.0, std=1.0)])

# Create train/test/validation dataset objects 
trainset = ADDataTrain(ad_dir = path + "/data/AD_NC/train/AD", 
                       nc_dir = path + "/data/AD_NC/train/NC",
                       ads = train_ads, ncs = train_ncs,
                       transform = transform)
testset = ADDataTest(ad_dir = path + "/data/AD_NC/test/AD", 
                     nc_dir = path + "/data/AD_NC/test/NC",
                     ads = test_ads, ncs = test_ncs,
                     transform = transform)
valset = ADDataTest(ad_dir = path + "/data/AD_NC/test/AD", 
                    nc_dir = path + "/data/AD_NC/test/NC",
                    ads = val_ads, ncs = val_ncs,
                    transform = transform)

# Load datasets to DataLoader 
train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=batch_size)


