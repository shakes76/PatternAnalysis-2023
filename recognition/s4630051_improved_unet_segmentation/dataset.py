from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

TEST = True

DATA_PATH = os.path.join('s4630051_improved_unet_segmentation', 'test_data' if TEST else 'data')

TRAIN_PATH = os.path.join(DATA_PATH, 'train', 'ISIC-2017_Training_Data')
TRAIN_SEG_PATH = os.path.join(DATA_PATH, 'train', 'ISIC-2017_Training_Part1_GroundTruth')

VALIDATE_PATH = os.path.join(DATA_PATH, 'validate', 'ISIC-2017_Validation_Data')
VALIDATE_SEG_PATH = os.path.join(DATA_PATH, 'validate', 'ISIC-2017_Validation_Part1_GroundTruth')

TEST_PATH = os.path.join(DATA_PATH, 'test', 'ISIC-2017_Test_v2_Data')
TEST_SEG_PATH = os.path.join(DATA_PATH, 'test', 'ISIC-2017_Test_v2_Part1_GroundTruth')

IMAGE_SIZE = 256

class ISICDataset(Dataset):
    """
    Dataset class for ISIC 2017 dataset.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Path to the image directory.
            mask_dir (string): Path to the mask directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Set image and mask directories
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Set transform
        self.transform = transform
        # Get image and mask names
        self.image_names = os.listdir(image_dir)
        self.mask_names = os.listdir(mask_dir)
        
        self.length = len(self.image_names)
        
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return self.length
    
    def __getitem__(self, idx):
        """
        Returns a sample of the dataset.
        """
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
                
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(mask)
            mask = transforms.ToTensor()(mask)
            
        return image, mask

# Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7084, 0.5822, 0.5361],
                                        std=[0.0948, 0.1099, 0.1240])
    ]),
    'validate': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7084, 0.5822, 0.5361],
                                        std=[0.0948, 0.1099, 0.1240])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7084, 0.5822, 0.5361],
                                        std=[0.0948, 0.1099, 0.1240])
    ])    
}

def get_datasets():
    """
    Returns the datasets
    """
    train_set = ISICDataset(TRAIN_PATH, TRAIN_SEG_PATH, transform=data_transforms['train'])
    validate_set = ISICDataset(VALIDATE_PATH, VALIDATE_SEG_PATH, transform=data_transforms['validate'])
    test_set = ISICDataset(TEST_PATH, TEST_SEG_PATH, transform=data_transforms['test'])

    return train_set, validate_set, test_set
