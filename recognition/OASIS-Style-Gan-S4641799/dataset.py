import utils, os
from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from PIL import Image

"""
Author name: Eli Cox
File name: dataset.py
Last modified: 22/11/2023
Data loader classes and image transformer for the StyleGAN model
"""

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)), # Resize image to match the size being trained on
    transforms.RandomHorizontalFlip(), # Randomly flip images to expand dataset artificially
    transforms.Grayscale(3), # Convert the data to greyscale in three channels as the model is expecting
    transforms.ToTensor(), # Convert the data into a tensor format
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalise the image data
])

# Directories storing the images
data_dir = "keras_png_slices_data/"
subfolder = "keras_png_slices_"
# Modifiers used in the data set for different data types
folder_suffix = "seg_"
folder_type = {"test", "train", "validate"}

"""
Collects image files to be used by the dataloader for the training, testing, and validation processes. 
"""
class FolderLoader(dataset.Dataset):
    def __init__(self, root, transform=None, includeSeg = False):
        self.root_dir = root
        self.transform = transform
        self.includeSeg = includeSeg
        self.image_paths = self._make_dataset()


    """
    Discovers each image in the required folders and stores their location in a list
    """
    def _make_dataset(self):
        image_paths = []
        directories = [self.root_dir]
        if self.includeSeg:
            # Include seg data types if requested
            directories.append(self.root_dir.replace('data/' + subfolder, 'data/' + subfolder+folder_suffix))
        for directory in directories:
            # Discover png images in the directories given
            for subdir, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".png"):
                        image_paths.append(os.path.join(subdir, file))
        return image_paths

    """
    Returns how many images were discovered and stored
    """
    def __len__(self):
        return len(self.image_paths)

    """
    Returns the data of an image located at the given index
    """
    def __getitem__(self, index):
        # REtrieve an item at the requested index
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        if self.transform:
            # Apply transformations to the requested image
            image = self.transform(image)
        return image

"""
Creates a data loader instance of the required dataset type
"""
def create_data_loader(dataset_type, transform_override = None):
    # Ensure we have access to the data type requested 
    assert dataset_type in folder_type
    if transform_override:
        # Return a dataloader with the requested transforms instead of the default
        dataset = FolderLoader(root=data_dir+subfolder+dataset_type, transform=transform_override, includeSeg=True)
    else:
        dataset = FolderLoader(root=data_dir+subfolder+dataset_type, transform=transform, includeSeg=True)
    #Generate a loader to encapsulate the data
    loader = DataLoader(dataset, batch_size=utils.BATCH_SIZE, shuffle=True)
    return loader
