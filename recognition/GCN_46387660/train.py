import torch
import torch.nn as nn
import modules
import dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# get model (GCN)
model = modules.GCN(in_channels=128, num_classes=4)
model = model.to(device)

# get dataset
dataset = dataset.data

print('starting test')

# need to train and then save to be able to use her for the predict file