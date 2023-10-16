import torch
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/'

data_transform = transforms.Compose([transforms.Resize((512, 512)),
                                             transforms.ToTensor()])

mask_transform = transforms.Compose([transforms.Resize((512, 512)),
                                            transforms.Grayscale(num_output_channels=1),  # Convert to single channel
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x > 0.5)  # Threshold to 0 or 1
                                            ])

# Training dataset
train_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1-2_Training_Input_x2', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

# Testing dataset
test_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1-2_Test_Input', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

# Truth dataset
truth_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1_Training_GroundTruth_x2', transform=mask_transform)
truth_loader = torch.utils.data.DataLoader(truth_set, batch_size=16, shuffle=True)

