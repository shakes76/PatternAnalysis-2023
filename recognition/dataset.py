from torchvision import transforms, datasets
import torch

path = "./ADNI/"
batch_size = 20

transformations = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomCrop(240), #should i center crop too
    transforms.ToTensor(), #might need to change the dimensions of the image
])



dataset = datasets.ImageFolder(path, transform=transformations) # datasets.CelebA(root=path, download=True, transform=transformations)
training_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
testing_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)