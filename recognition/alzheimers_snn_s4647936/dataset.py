import os
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Generate triplets for training Siamese networks using triplet loss.
    
    For each anchor image from either the AD or NC class, a positive image is selected from 
    the same patient but a different slice. A negative image is then chosen from the opposite class.
    
    Args:
        root_dir (str): Root directory containing AD and NC image subdirectories.
        mode (str): Either 'train' or 'test'.
        transform (callable, optional): Transformations applied to the images.
    
    Returns:
        tuple: A triplet of images - (anchor, positive, negative).
    """
    
    def __init__(self, root_dir, mode='train', transform=None, split_ratio=0.8):
        self.root_dir = root_dir # root_dir = "/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC"
        self.mode = mode
        self.transform = transform
        
        # Directories for AD and NC images
        self.ad_dir = os.path.join(root_dir, mode, 'AD')
        self.nc_dir = os.path.join(root_dir, mode, 'NC')

        # Load all image paths
        self.ad_paths = [os.path.join(self.ad_dir, img) for img in os.listdir(self.ad_dir)]
        self.nc_paths = [os.path.join(self.nc_dir, img) for img in os.listdir(self.nc_dir)]

        train_ad_paths, test_ad_paths, train_nc_paths, test_nc_paths = patient_wise_split(self.ad_paths, self.nc_paths, split_ratio)

        if mode == 'train':
            self.ad_paths = train_ad_paths
            self.nc_paths = train_nc_paths

            # Integrate data augmentation if in training mode
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transform
            ])

        elif mode == 'test':
            self.ad_paths = test_ad_paths
            self.nc_paths = test_nc_paths
            print("Sample AD paths:", self.ad_paths[:5])
            print("Sample NC paths:", self.nc_paths[:5])


    def __len__(self):
        return len(self.ad_paths) + len(self.nc_paths)  # combined length

    def __getitem__(self, idx):
        # Decide whether to take AD or NC as anchor based on index
        if idx < len(self.ad_paths):
            anchor_path = self.ad_paths[idx]
            positive_paths = self.ad_paths
            negative_paths = self.nc_paths
        else:
            anchor_path = self.nc_paths[idx - len(self.ad_paths)]  # offset by length of ad_paths
            positive_paths = self.nc_paths
            negative_paths = self.ad_paths

        # Extract patient ID from the filename
        patient_id = os.path.basename(anchor_path).split('_')[0]

        # Choose a positive image from the same patient
        positive_path = random.choice([path for path in positive_paths if os.path.basename(path) != os.path.basename(anchor_path) and patient_id in os.path.basename(path)])
        
        # Choose a negative image from a different patient
        negative_path = random.choice([path for path in negative_paths if patient_id not in os.path.basename(path)])
        anchor_image = Image.open(anchor_path)
        positive_image = Image.open(positive_path)
        negative_image = Image.open(negative_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        # Decide label based on anchor image path
        label = 0 if "/AD/" in anchor_path else 1

        return anchor_image, positive_image, negative_image, label
    

def patient_wise_split(ad_paths, nc_paths, split_ratio=0.8):
    """
    Split the AD and NC data patient-wise.
    
    Args:
    - ad_paths: List of paths to AD images.
    - nc_paths: List of paths to NC images.
    - split_ratio: Proportion of data to use for training.
    
    Returns:
    - train_ad_paths: List of AD training paths.
    - test_ad_paths: List of AD testing paths.
    - train_nc_paths: List of NC training paths.
    - test_nc_paths: List of NC testing paths.
    """

    # Extract patient IDs
    ad_patient_ids = list(set(os.path.basename(path).split('_')[0] for path in ad_paths))
    nc_patient_ids = list(set(os.path.basename(path).split('_')[0] for path in nc_paths))

    # Split patient IDs for training and testing
    train_ad_ids = random.sample(ad_patient_ids, int(split_ratio * len(ad_patient_ids)))
    train_nc_ids = random.sample(nc_patient_ids, int(split_ratio * len(nc_patient_ids)))

    test_ad_ids = list(set(ad_patient_ids) - set(train_ad_ids))
    test_nc_ids = list(set(nc_patient_ids) - set(train_nc_ids))

    # Get paths based on split IDs
    train_ad_paths = [path for path in ad_paths if os.path.basename(path).split('_')[0] in train_ad_ids]
    test_ad_paths = [path for path in ad_paths if os.path.basename(path).split('_')[0] in test_ad_ids]

    train_nc_paths = [path for path in nc_paths if os.path.basename(path).split('_')[0] in train_nc_ids]
    test_nc_paths = [path for path in nc_paths if os.path.basename(path).split('_')[0] in test_nc_ids]

    return train_ad_paths, test_ad_paths, train_nc_paths, test_nc_paths

