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
class PatchEmbedding(nn.Module):
    """Takes a 2D input image and splits it into fixed-sized patches and linearly embeds each of them.

    Changes the dimensions from H x W x C to N x (P^2 * C), where 
    (H, W, C) is the height, width, number of channels of the image,
    N is the number of patches, 
    and P is the dimension of each patch; P^2 represents a flattened patch.
    """
    def __init__(self, ngpu):
        super(PatchEmbedding, self).__init__()
        self.ngpu = ngpu
        # Puts image through Conv2D layer with kernel_size = stride to ensure no patches overlap.
        # This will split image into fixed-sized patches; each patch has the same dimensions
        # Then, each patch is flattened, including all channels for each patch.
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=num_channels,
                        out_channels=embed_dim,
                        kernel_size=patch_size,
                        stride=patch_size,
                        padding=0),
            nn.Flatten(start_dim=2, end_dim=3)
        )

    def forward(self, input):
        return self.main(input).permute(0, 2, 1)  # Reorder the dimensions


class ViT(nn.Module):
    """Creates a vision transformer model."""
    def __init__(self, ngpu):
        super(ViT, self).__init__()
        self.ngpu = ngpu

        self.patch_embedding = PatchEmbedding(workers)
        self.prepend_embed_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.position_embed_token = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)
    
    def forward(self, input):
        prepend_embed_token_expanded = self.prepend_embed_token.expand(batch_size, -1, -1)

        input = self.patch_embedding(input)
        input = torch.cat((prepend_embed_token_expanded, input), dim=1)
        input = input + self.position_embed_token


def imshow(img):
    img = img / 2 + 0.5  # Unnormalize (assuming your normalization was (0.5, 0.5, 0.5))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    images = images[:10]
    labels = labels[:10]

    #imshow(torchvision.utils.make_grid(images, nrow=5))

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
    #imshow(image)
    #plt.axis(False)
    image_patched = patch_embedding(image.unsqueeze(0))  # Run conv layer through image
    print(image_patched.shape)  # Should have dimensions (embed size, sqrt(num_patches), sqrt(num_patches))

    # Single feature map in tensor form
    single_feature_map = image_patched[:, 0, :, :]
    print(single_feature_map, single_feature_map.requires_grad)

    # Dimension 2 sqrt(num_patches) -> height of num_patches, dimension 3 is sqrt(num_patches) -> width of num_patches
    flatten = nn.Flatten(start_dim=2, end_dim=3)
    flattened_image_patched = flatten(image_patched)
    print(f"Flattened: {flattened_image_patched.shape}")  # Should be embed size, num_patches

    reshaped_flattened_image_patched = flattened_image_patched.permute(0, 2, 1)  # Need to swap embed size and num_patches order
    # This achieves the resizing in the paper: H x W x C -> N x (P^2*C)
    print(f"Reshaped:{reshaped_flattened_image_patched.shape}")

    # Test Patching module on random tensor
    random_image_tensor = torch.randn(1, 3, 224, 224)
    patch_embedding = PatchEmbedding(workers)
    patch_embedding_output = patch_embedding(random_image_tensor)
    print(f"In shape: {random_image_tensor.shape}")
    print(f"Out shape: {patch_embedding_output.shape}")
    print({patch_embedding_output})

    # Need to prepend a learnable embedding to the sequence of embeded patches.
    embed_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
    prepended_patch_embedding = torch.cat((embed_token, patch_embedding_output), dim=1)  
    print(f"Prepended embedding: {prepended_patch_embedding}")

    # Need to add position embedding; E_pos, where E_pos has dimensions (N+1) x D
    # Used to retain positional information of the patches.
    positional_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)
    print(f"Positional embed shape: {positional_embed.shape}, Current patch shape: {prepended_patch_embedding.shape}")
    print(f"Positional embed tensor: {positional_embed}")

    patch_and_position_embedding = prepended_patch_embedding + positional_embed
    print(f"Final: {patch_and_position_embedding}")
    print(f"Final shape: {patch_and_position_embedding.shape}")   

def main():
    test


if __name__ == '__main__':
    main()