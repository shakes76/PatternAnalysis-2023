from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os

BATCH_SIZE = 10

# custom dataset
class ISICDataset(Dataset):
    def __init__(self, img_dir, transform, truth_dir='', split_ratio=0.8, train=True):
        super(ISICDataset, self).__init__()
        self.img_dir = img_dir
        self.image_files = sorted(os.listdir(img_dir))

        self.image_files.remove("ATTRIBUTION.txt")
        self.image_files.remove("LICENSE.txt")
        
        self.truth_dir = truth_dir
        self.truth_files = sorted(os.listdir(truth_dir))

        self.truth_files.remove("ATTRIBUTION.txt")
        self.truth_files.remove("LICENSE.txt")

        self.transform = transform

        total_samples = len(self.image_files)
        split_idx = int(split_ratio * total_samples)

        if train:
            self.image_files = self.image_files[:split_idx]
            self.truth_files = self.truth_files[:split_idx]
        else:
            self.image_files = self.image_files[split_idx:]
            self.truth_files = self.truth_files[split_idx:]

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

# transforms to be put on the images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((512, 512), antialias=True)])

# create datasets
train_data = ISICDataset(img_dir="data/train_data", truth_dir="data/train_truth", transform=transform, train=True)
val_data = ISICDataset(img_dir="data/train_data", truth_dir="data/train_truth", transform=transform, train=False)

# create dataloaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

