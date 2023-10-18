import torch
import torch.nn as nn
import torch.optim as optim
from modules import ImprovedUNet
from dataset import UNetData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/'
    
# Hyper-parameters
num_epochs = 1
learning_rate = 0.1
image_height = 512 
image_width = 512

def main():
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)

    data = UNetData(path=path, height=image_height, width=image_width)

if __name__ == "__main__":
    main()