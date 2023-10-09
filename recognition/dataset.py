from torchvision import transforms, datasets
import torch

path = "./ADNI/"
batch_size = 20

class ADNI():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transformations = transforms.Compose([transforms.Grayscale(1), transforms.RandomCrop(240), transforms.ToTensor()])
        self.dataset = datasets.ImageFolder(path, transform=self.transformations) # datasets.CelebA(root=path, download=True, transform=transformations)
        self.training_data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.testing_data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        