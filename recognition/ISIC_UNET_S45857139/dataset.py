import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ISICDataset(Dataset):
    """
    Custom dataset for loading ISIC skin lesion images and their corresponding masks.

    Args:
        - image_dir (str): Path to the directory with image files.
        - mask_dir (str): Path to the directory with mask files.
        - transform (callable, optional): Optional transform to be applied on both image and mask. Defaults to None.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Exclude files with '_segmentation.png' in their names from image files list
        self.image_files = [f for f in os.listdir(image_dir) if '_segmentation.png' not in f]
        
    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.image_files)
        
    def __getitem__(self, idx):
        """
        Fetches the image and mask at the specified index.
        
        Args:
            - idx (int): Index of the item to retrieve.

        Returns:
            - tuple: (image, mask) pair.
        """
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.image_files[idx].replace('.jpg', '_segmentation.png'))
        
        image = Image.open(img_name)
        mask = Image.open(mask_name)  

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def get_isic_dataloader(image_dir, mask_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Helper function to create a DataLoader for the ISIC dataset.

    Args:
        - image_dir (str): Path to the directory with image files.
        - mask_dir (str): Path to the directory with mask files.
        - batch_size (int, optional): Number of samples per batch. Defaults to 32.
        - shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        - num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
        - DataLoader: DataLoader object for the ISIC dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = ISICDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
