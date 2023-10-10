import torch
import torchvision

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/COMP3710/Practicals/Prac_2/'

# Training dataset
train_set = torchvision.datasets.ImageFolder(root=path+'train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

# Testing dataset
test_set = torchvision.datasets.ImageFolder(root=path+'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

# Validation dataset
validate_set = torchvision.datasets.ImageFolder(root=path+'validate')
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size=16, shuffle=True)