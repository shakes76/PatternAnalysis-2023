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

batch_size = 64
workers = 2

# Images are 256 by 240 pixels. Resize them to 224 by 224; must be divisible by 16
image_size = 224  # Resized 2D image input
patch_size = 16  # Dimension of a patch
num_patches = (image_size // patch_size) ** 2  # Number of patches in total
num_channels = 3  # 3 channels for RGB
embed_dim = 768  # Hidden size D of ViT-Base model from paper, equal to [(patch_size ** 2) * num_channels]

# Create the dataset
dataroot = "AD_NC"
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                        ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# ------------------------------------------------------------------
# Patch Embedding


#class PatchEmbedding(nn.Module):


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

    imshow(torchvision.utils.make_grid(images, nrow=5))

    # Print labels
    print(' '.join('%5s' % labels[j].item() for j in range(10)))

    # Get initial shape, should be 224 (H) by 224 (W) by 3 (C)
    sample_datapoint = torch.unsqueeze(dataset[0][0], 0)
    print("Initial shape: ", sample_datapoint.shape)

    # Test patch embedding for 1 image
    # For a 224 by 224 image with 16 patch size, this gives us a 14 by 14 number of patches, 
    # each patch having 16 by 16 dimensions and 3 channels
    patch_embedding = nn.Conv2d(in_channels=num_channels,
                                out_channels=embed_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0)
    image = dataset[0][0]
    imshow(image)
    plt.axis(False)
    image_patched = patch_embedding(image.unsqueeze(0))  # Run conv layer through image
    print(image_patched.shape)  # Should have dimensions (embed size, sqrt(num_patches), sqrt(num_patches))

    # Single feature map in tensor form
    single_feature_map = image_patched[:, 0, :, :]
    print(single_feature_map, single_feature_map.requires_grad)

    # Dimension 2 sqrt(num_patches) -> height of num_patches, dimension 3 is sqrt(num_patches) -> width of num_patches
    flatten = nn.Flatten(start_dim=2, end_dim=3)
    flattened_image_patched = flatten(image_patched)
    print(f"Flattened: {flattened_image_patched.shape}")  # Should be embed size, num_patches

if __name__ == '__main__':
    main()