# imports
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Parameters
batchSize = 64

# transforms
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = ImageFolder("/home/groups/comp3710/ADNI/AD_NC/train", transform=transform)
testset = ImageFolder("/home/groups/comp3710/ADNI/AD_NC/test", transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

# constant for classes
classes = ('AD', 'NC')
channels = 1