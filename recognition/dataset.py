from torchvision import transforms, datasets
import torch

path = "./ADNI/AD_NC/"
batch_size = 20


#Need to change the resize method
# randaugment, centercropping, Normalize (0-1)
#change randomcrop to centercrop
class ADNI():
    def __init__(self, batch_size):
        self.batch_size = batch_size #remove the normalise
        self.transformations = transforms.Compose([transforms.Grayscale(1), transforms.CenterCrop(240), transforms.ToTensor()])
        self.dataset = datasets.ImageFolder(path + "train", transform=self.transformations) # datasets.CelebA(root=path, download=True, transform=transformations)
        self.training_data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.testing_data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
