import torch
import torchvision

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/'

# Training dataset
train_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1-2_Training_Input_x2')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False)

# Testing dataset
test_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1-2_Test_Input')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

# Truth dataset
truth_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1_Training_GroundTruth_x2')
truth_loader = torch.utils.data.DataLoader(truth_set, batch_size=16, shuffle=False)