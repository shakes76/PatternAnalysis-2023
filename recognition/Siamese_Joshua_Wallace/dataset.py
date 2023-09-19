import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


"""

"""
class Dataset():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, batch_size = 32, root_dir = './AD_NC') :
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.train_loader = None
        self.test_loader = None
    
    def load_train(self) -> None:
        train_dataset = ImageFolder(root=f"{self.root_dir}/train", transform=self.transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_train(self) -> DataLoader :
        if not self.train_loader :
            print('Retrieving trainset.')
            self.load_train()
        return self.train_loader

    def load_test(self) -> None :
        test_dataset = ImageFolder(root=f"{self.root_dir}/test", transform=self.transform)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_test(self) -> DataLoader :
        if not self.test_loader :
            print('Retrieving testset.')
            self.load_test()
        return self.test_loader