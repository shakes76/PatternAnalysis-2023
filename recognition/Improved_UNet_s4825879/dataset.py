from torch.utils.data import Dataset
from PIL import Image
import os

# custom dataset
class ISICDataset(Dataset):
    def __init__(self, img_dir, transform):
        super(ISICDataset, self).__init__()
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)
        self.transform = transform

    # function returns the length of the dataset
    def __len__(self):
        return len(self.image_files)

    # get specific item from dataset
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
