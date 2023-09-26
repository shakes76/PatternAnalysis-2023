import torch

from dataset import *
from modules import *
import torchvision.transforms as transforms

#-------------
# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Warning, CUDA not found. Using CPU.")
print()


#---------------
# Hyper Parameters
learning_rate = 0.001
num_epochs = 4
path = r"c:\Users\rotax\OneDrive\Desktop\COMP3710\AD_NC"


#-----------------
# Data
batch_size = 32

train_path = path + "\train\AD"
test_path = path + "\test\AD"

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = ImageDataset(directory=train_path,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_data = ImageDataset(directory=test_path,
                         transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)