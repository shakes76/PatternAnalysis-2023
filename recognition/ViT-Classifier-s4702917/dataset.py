# imports
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder

# Parameters
batchSize = 64

# transforms
class SquarePad:
	def __call__(self, image: torch.Tensor):
		_, w, h = image.shape
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

transform = transforms.Compose(
	[transforms.ToTensor(),
	SquarePad(),
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