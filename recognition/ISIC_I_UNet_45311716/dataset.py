import torch
import torchvision
import torchvision.transforms as transforms

class UNetData():
    def __init__(self, path, height, width, batch):

        self.path = path
        self.batch = batch
        # RGB 3 channel input images
        self.data_transform = transforms.Compose([transforms.Resize((height, width)),
                                            transforms.ToTensor()])
        # Black and white 1 channel mask image
        self.mask_transform = transforms.Compose([transforms.Resize((height, width)),
                                            transforms.Grayscale(num_output_channels=1),  # Convert to single channel (grayscale)
                                            transforms.ToTensor(),  # Convert to a tensor
                                            transforms.Lambda(lambda x: x > 0.5)  # Threshold to 0 or 1
                                            ])
        
    def get_train_loader(self):    
        # Training dataset
        train_set = torchvision.datasets.ImageFolder(root=self.path+'training_images', transform=self.data_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch, shuffle=True)

        # Truth dataset
        truth_set = torchvision.datasets.ImageFolder(root=self.path+'training_masks', transform=self.mask_transform)
        truth_loader = torch.utils.data.DataLoader(truth_set, batch_size=self.batch, shuffle=True)

        train_data = []
        
        print(' - - Filling Train - - ')
        for x, y in zip(train_loader, truth_loader):
            train_data.append((x[0], y[0]))
        
        return train_data


    def get_valid_loader(self): 
        # Validating dataset
        valid_set = torchvision.datasets.ImageFolder(root=self.path+'validation_images', transform=self.data_transform)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=self.batch, shuffle=True)

        # Truth dataset
        valid_t_set = torchvision.datasets.ImageFolder(root=self.path+'validation_masks', transform=self.mask_transform)
        valid_t_loader = torch.utils.data.DataLoader(valid_t_set, batch_size=self.batch, shuffle=True)

        valid_data = []
        
        print(' - - Filling Valid - - ')
        for x, y in zip(valid_loader, valid_t_loader):
            valid_data.append((x[0], y[0]))
    
        print(' - - Data Loaded - - ')

        return valid_data

    def get_test_loader(self):    
        # Testing dataset
        test_set = torchvision.datasets.ImageFolder(root=self.path+'testing_images', transform=self.data_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch, shuffle=True)

        # Truth dataset
        test_t_set = torchvision.datasets.ImageFolder(root=self.path+'testing_masks', transform=self.mask_transform)
        test_t_loader = torch.utils.data.DataLoader(test_t_set, batch_size=self.batch, shuffle=True)

        test_data = []
        
        print(' - - Filling Test - - ')
        for x, y in zip(test_loader, test_t_loader):
            test_data.append((x[0], y[0]))
        
        return test_data
