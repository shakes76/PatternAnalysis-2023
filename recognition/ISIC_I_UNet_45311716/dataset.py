import torch
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

class UNetData():
    def __init__(self, path, height, width):

        # RGB 3 channel input images
        data_transform = transforms.Compose([transforms.Resize((height, width)),
                                                    transforms.ToTensor()])

        # Black and white 1 channel mask image
        mask_transform = transforms.Compose([transforms.Resize((height, width)),
                                                    transforms.Grayscale(num_output_channels=1),  # Convert to single channel
                                                    transforms.ToTensor(),
                                                    transforms.Lambda(lambda x: x > 0.5)  # Threshold to 0 or 1
                                                    ])

        # Training dataset
        train_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1-2_Training_Input_x2', transform=data_transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)

        # Testing dataset
        test_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1-2_Test_Input', transform=data_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)

        # Truth dataset
        truth_set = torchvision.datasets.ImageFolder(root=path+'ISIC2018_Task1_Training_GroundTruth_x2', transform=mask_transform)
        self.truth_loader = torch.utils.data.DataLoader(truth_set, batch_size=16, shuffle=True)

