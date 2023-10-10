""" Data loader for loading and preprocessing the dataset. """
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class OASISBrainDataset(Dataset):
    """OASIS Brain dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_fir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
                                            Defaults to None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    
def create_dataloader(root_dir, batch_size=32, num_workers=4):
    # Define the transforms to be applied to the images
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset
    dataset = OASISBrainDataset(root_dir=root_dir, transform=data_transform)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

def show_image(tensor_image):
    """Show image from tensor."""
    image = tensor_image.permute(1, 2, 0)
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    dataloader = create_dataloader("D:\keras_png_slices_data\keras_png_slices_data\keras_png_slices_test")
    
    images = next(iter(dataloader))

    show_image(images[0])
    print(images[0].shape)