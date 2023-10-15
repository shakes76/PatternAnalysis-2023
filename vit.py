import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

manualSeed = 999
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

# Images are 256 by 240 pixels. Resize them to 224 by 224; must be divisible by 16
image_size = 224
num_patches = 16
num_channels = 3
batch_size = 128
workers = 2
# Create the dataset
dataroot = "AD_NC"
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip()
                        ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize (assuming your normalization was (0.5, 0.5, 0.5))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    images = images[:10]
    labels = labels[:10]

    imshow(torchvision.utils.make_grid(images))

    # Print labels
    print(' '.join('%5s' % labels[j].item() for j in range(10)))

if __name__ == '__main__':
    main()