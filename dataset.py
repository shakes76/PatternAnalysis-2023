from torch.utils.data import Dataset
from PIL import Image
import os
from utils import display_images_from_directory


directory_path = 'C:\\Users\\25060\\Desktop\\ISIC-2017_Training_Part1_GroundTruth'
display_images_from_directory(directory_path)

class ISICDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_exts=('.jpg', '.jpeg', '.png')):
        self.root_dir = root_dir
        self.transform = transform
        
        # Filter images by extensions
        self.image_files = [f for f in sorted(os.listdir(root_dir)) if f.endswith(image_exts)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Ensure RGB

        # Assuming masks are in a 'masks' subfolder and have the same names as images
        mask_name = os.path.join(self.root_dir, 'masks', self.image_files[idx])
        mask = Image.open(mask_name).convert('L')  # Load mask as grayscale

        if self.transform:
            image = self.transform(image)
            # Ensure mask transform doesn't involve color jitter, normalization, etc.
            # For simplicity, applying the same transform here, but it may need adjustments.
            mask = self.transform(mask)

        return image, mask
