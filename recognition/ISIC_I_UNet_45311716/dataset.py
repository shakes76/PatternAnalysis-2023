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
                                            transforms.Grayscale(num_output_channels=1),  # Convert to single channel (grayscale)
                                            transforms.ToTensor(),  # Convert to a tensor
                                            transforms.Lambda(lambda x: x > 0.5)  # Threshold to 0 or 1
                                            ])
        # Training dataset
        train_set = torchvision.datasets.ImageFolder(root=path+'training_images', transform=data_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False)

        # Truth dataset
        truth_set = torchvision.datasets.ImageFolder(root=path+'training_masks', transform=mask_transform)
        truth_loader = torch.utils.data.DataLoader(truth_set, batch_size=16, shuffle=False)

        self.train_data = []
        
        print(' - - Filling Train - - ')
        for x, y in zip(train_loader, truth_loader):
            self.train_data.append((x[0], y[0]))

        # ---------------------------------------------------------

        # Validating dataset
        valid_set = torchvision.datasets.ImageFolder(root=path+'validation_images', transform=data_transform)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=16, shuffle=False)

        # Truth dataset
        valid_t_set = torchvision.datasets.ImageFolder(root=path+'validation_masks', transform=mask_transform)
        valid_t_loader = torch.utils.data.DataLoader(valid_t_set, batch_size=16, shuffle=False)

        self.valid_data = []
        
        print(' - - Filling Valid - - ')
        for x, y in zip(valid_loader, valid_t_loader):
            self.valid_data.append((x[0], y[0]))
    
        print(' - - Data Loaded - - ')

