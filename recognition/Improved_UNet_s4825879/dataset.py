from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import DataTransform
from PIL import Image
import os

# specify batch size
BATCH_SIZE = 3

# Specify vished image size
IMAGE_SIZE = 64

# Specify dataset directory path
TRAIN_DATA_PATH = 'data/train_data'
TRAIN_TRUTH_PATH = 'data/train_truth'
TEST_TRUTH_PATH = TRAIN_TRUTH_PATH
TEST_DATA_PATH = TRAIN_DATA_PATH

# custom dataset
class ISICDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 transform, 
                 truth_dir='', 
                 split_ratio=0.8, 
                 train=True,
                 ):
        super(ISICDataset, self).__init__()
        self.train = train
        
        self.img_dir = img_dir
        self.image_files = sorted(os.listdir(img_dir))
        
        self.truth_dir = truth_dir
        self.truth_files = sorted(os.listdir(truth_dir))

        self.transform = transform

        total_samples = len(self.image_files)
        split_idx = int(split_ratio * total_samples)

        # remove all files that are not images from the dataset
        for file in self.image_files:
            if not file.endswith('.jpg'):
                self.image_files.remove(file)
        for file in self.truth_files:
            if not file.endswith('.png'):
                self.truth_files.remove(file)

        self.train = train
        
        # split directory into seperate training and validation sets
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
            if self.train:
                (image, truth) = self.transform((image, truth))
            else:
                image = self.transform(image)
                truth = self.transform(truth)
        return image, truth

# custom data transformations
train_transform = DataTransform(size=(IMAGE_SIZE,IMAGE_SIZE))
# transforms for validation data
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)])

# create datasets
train_data = ISICDataset(img_dir=TRAIN_DATA_PATH, truth_dir=TRAIN_TRUTH_PATH ,split_ratio=0.9, transform=train_transform, train=True)
val_data = ISICDataset(img_dir=TRAIN_DATA_PATH, truth_dir=TRAIN_TRUTH_PATH, split_ratio=0.9,transform=transform, train=False)
test_data = ISICDataset(img_dir=TEST_DATA_PATH, truth_dir=TEST_TRUTH_PATH, split_ratio=0.0, transform=transform, train=False)

# create dataloaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

