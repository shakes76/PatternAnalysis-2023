from torch.utils.data import Dataset
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None):
        """
        Args:
 
            subset (string): 'train' or 'test' to specify which subset of the data to use.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.

        """
        self.transform = transform
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        # build a list of all the file paths
        self.data_paths = []
        for label_class in ['AD', 'NC']:
            class_dir = os.path.join(root_dir, self.subset, label_class)
            for filename in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, filename)):
                    self.data_paths.append((os.path.join(class_dir, filename), label_class))

       

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path, label_class = self.data_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale

        label = 1 if label_class == 'AD' else 0  # 1 for Alzheimer's (AC), 0 for normal control (NC)

        if self.transform:
            image = self.transform(image)

        return image, label

