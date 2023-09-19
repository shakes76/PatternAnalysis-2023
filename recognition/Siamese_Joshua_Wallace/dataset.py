import torch
from torchvision import datasets, transforms



class Dataset():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    def __init__(self, batch_size = 32, root_dir = './AD_NC') :
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.train_loader = None
        self.test_loader = None
    
    def load_train(self) :
        train_dataset = datasets.ImageFolder(root=f"{self.root_dir}/train", transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_train(self) :
        if not self.train_loader :
            print('Retrieving trainset.')
            self.load_train()
        return self.train_loader

    def load_test(self) :
        test_dataset = datasets.ImageFolder(root=f"{self.root_dir}/test", transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_test(self) :
        if not self.test_loader :
            print('Retrieving testset.')
            self.load_test()
        return self.test_loader