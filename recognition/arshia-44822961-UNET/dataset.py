import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Global constants for the dimensions to which images and masks will be resized
TRANSFORMED_X = 256  # width after resizing
TRANSFORMED_Y= 256  # height after resizing

"""
    PyTorch Dataset class for loading ISIC melanoma detection dataset.

    Parameters:
    - img_path (str): Path to the directory containing image files.
    - mask_path (str): Path to the directory containing mask files.
    - transform (call): Transform to be applied on the images and masks.

    Methods:
    - __len__(): Returns the total number of images in the dataset.
    - __getitem__(idx): Returns the image and its corresponding mask at the given index `idx`.
    - match_mask_to_image(img_filename): Converts image filename to its corresponding mask filename.
    """
class ISICDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_filenames = sorted(os.listdir(img_path))
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def match_mask_to_image(self, img_filename):
    # Convert the image filename to its corresponding mask filename
        base_name = os.path.splitext(img_filename)[0]  # This removes the .jpg or any extension
        return base_name + '_segmentation.png'

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.img_filenames[idx])
        mask_name = os.path.join(self.mask_path, self.match_mask_to_image(self.img_filenames[idx]))

        img = Image.open(img_name).convert('RGB')  # Convert to RGB
        mask = Image.open(mask_name).convert('L')  # Convert to grayscale for segmentation masks. 

        img = img.resize((TRANSFORMED_Y, TRANSFORMED_X))
        mask = mask.resize((TRANSFORMED_Y, TRANSFORMED_X), Image.NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0  # Convert to single channel tensor

        return img, mask

# TODO - remove and place in main method or train.py
# Define transforms for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # remove this temporarily 
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


"""
Helper method to compute mean and standard deviations for the train dataset.
"""
def compute_mean_std(loader):
    mean = 0.0
    squared_mean = 0.0
    total_samples = 0.0
    
    for images, _ in loader:
        images_flat = images.view(images.size(0), images.size(1), -1)
        mean += images_flat.mean(2).sum(0)
        squared_mean += (images_flat ** 2).mean(2).sum(0)
        total_samples += images.size(0)

    mean /= total_samples
    squared_mean /= total_samples
    std = (squared_mean - mean**2)**0.5

    return mean, std

# Main method - will not be in final dataset. 
# This is to compute mean and std deviations and also make sure data loading is working.

if __name__ == "__main__":
    print("hello")
    print(torch.version.cuda)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Hello world")    

    print("okay now to load in data. ")
    print(os.getcwd())

    file_path = "/home/Student/s4482296/report1/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/ISIC_0000000.jpg" # replace with your file path
    print(file_path)
    if os.path.isfile(file_path):
        print("okay")
    else:
        print("File not found.")


    train_directory = "/home/Student/s4482296/report1/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
    train_ground_truth_directory = "/home/Student/s4482296/report1/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"

    train_dataset = ISICDataset(train_directory,train_ground_truth_directory, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    mean, std = compute_mean_std(train_loader)

    # three values for RGB values as well.
    # gosh gpt 4 is really so smart. 
    print("Mean and standard deviation")
    print(mean)
    print(std)


# mean and standard deviation. 
# tensor([0.7084, 0.5821, 0.5360])
#tensor([0.1561, 0.1644, 0.1795])