import os
from PIL import Image
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    """
    Dataset class for the ISIC skin lesion dataset.

    Args:
        image_dir (str): Directory with all the images.
        mask_dir (str): Directory with all the masks.
        image_ext (str): File extension for the images.
        mask_ext (str): File extension for the masks.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, image_dir, mask_dir, image_ext='.jpg', mask_ext='_mask.jpg', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(image_ext)]
        self.mask_files = [f.replace(image_ext, mask_ext) for f in self.image_files]
        self.transform = transform

        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError("Image or mask directory does not exist.")

        if not self.image_files:
            raise FileNotFoundError("No image files found in the directory.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except IOError as e:
            raise IOError(f"Error opening image or mask file: {e}")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

