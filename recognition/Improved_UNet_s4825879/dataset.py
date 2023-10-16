from torch.utils.data import Dataset
from PIL import Image
import os

# custom dataset
class ISICDataset(Dataset):
    def __init__(self, img_dir, truth_dir, transform):
        super(ISICDataset, self).__init__()
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)

        self.image_files.remove("ATTRIBUTION.txt")
        self.image_files.remove("LICENSE.txt")

        self.truth_dir = truth_dir
        self.truth_files = os.listdir(truth_dir)

        self.truth_files.remove("ATTRIBUTION.txt")
        self.truth_files.remove("LICENSE.txt")

        self.transform = transform

    # function returns the length of the dataset
    def __len__(self):
        return len(self.image_files)

    # get specific item from dataset
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path)
        truth_path = os.path.join(self.truth_dir, self.truth_files[idx])
        truth = Image.open(truth_path)

        if self.transform:
            image = self.transform(image)
            truth = self.transform(truth)

        return image, truth
