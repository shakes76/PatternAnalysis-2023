import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm
from typing import List
import os

#from sklearn.utils import shuffle

KERAS_PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices'
BATCH_SIZE = 12

def to_channels(arr : np.ndarray , dtype = np.uint8 ) -> np.ndarray :
    '''
    Add number of channels as a new dimension. The output will be
    (n_channels, height, width) to fit to the UNet expectation.
    '''
    channels = [0, 1, 2, 3, 4, 5]
    res = np.zeros (arr.shape + ( len(channels),), dtype = dtype)
    for c in channels:
        c = int(c)
        res [..., c:c+1][arr == c] = 1
    #res = np.transpose(res, (2, 0, 1))
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    early_stop : Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # Initialize arrays based on the first image to get size information
    num = len(imageNames)
    example_image = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(example_image.shape) == 3:
        example_image = example_image[:, :, 0]  # Remove extra dimension if present

    if categorical:
        example_image = to_channels(example_image, dtype=dtype)
        channels, rows, cols = example_image.shape
        images = np.zeros((num, channels, rows, cols), dtype=dtype)
    else:
        rows, cols = example_image.shape
        images = np.zeros((num, 1, rows, cols), dtype=dtype)  # Initialize with a single channel

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Remove extra dimensions if present
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage  # Assign the data to the pre-allocated array
        else:
            images[i, 0, :, :] = inImage  # Ensure consistent shape for non-categorical data

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images  # Return 4D data [number of images, channels, rows(height), cols(width)]


def data_generator(image_files, label_files, batch_size):
    '''
    As DataLoader is not applicable for different shape of images and labels,
    data_generator works instead.
    Parameters:
    image_files: Image file paths
    label_files: Segmented image file paths
    batch_size: batch
    '''
    num_samples = len(image_files)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        image_batch = [image_files[i] for i in batch_indices]  # Add a new dimension for 1 channel
        label_batch = [label_files[i] for i in batch_indices]

        image_batch = torch.tensor(np.stack(image_batch), dtype=torch.float32)
        label_batch = torch.tensor(np.stack(label_batch), dtype=torch.float32)

        yield image_batch, label_batch

class CustomImageDataset(Dataset):
    '''
    Class to define the images in specified path as well as their segmented version.
    '''
    def __init__(self, img_dir: str, img_type: str, transform=None):
        """
        Prepare the image paths and load data from nbi library.
        Parameters:
        img_dir: Image directory
        transform: Transformations to apply.
        Return:
        3D tensors (Images, Labels)
        (n_channels, height, width)
        """
        self.img_dir = img_dir #Image directory.
        self.img_type = img_type #Train, validate or test.
        self.transform = transform #The transform method
        self.img_files = list()
        self.seg_img_files = list()
        data_list = os.listdir(self.img_dir + self.img_type[:-1])
        #Filenames fetched later
        self.img_files = [self.img_dir + self.img_type + data for data in data_list]
        data_list_seg = os.listdir(self.img_dir + '_seg' + self.img_type[:-1])
        self.seg_img_files = [self.img_dir + '_seg' + self.img_type + data for data in data_list_seg]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        #We return segmented image as label.
        
        img = load_data_2D([self.img_files[idx]], normImage=True)[0] #256x128
        img = np.transpose(img, (0, 1, 2))
        label = load_data_2D([self.seg_img_files[idx]], categorical=True)[0] #256x128
        if self.transform:
            image = self.transform(img)
            label = self.transform(label)
        print(image.shape, label.shape)
        return image, label
        '''
        img_path = self.img_files[idx]
        niftiImage = nib.load(img_path) #Load image from zs file.
        inImage = niftiImage.get_fdata(caching='unchanged') #Pixels in ndarray
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        label = self.seg_img_files[idx]
        if not label:
            label = to_channels(inImage)
        if self.transform:
            image = self.transform(inImage)
            label = self.transform(label)
        
        return image, label
        '''

def keras_dataloader(image_size: int, data_type: str):
    '''
    Create DataLoader for the CustomImageDataset defined above.
    Run this function three times for train, validation and test set.
    Returns both DataLoader and Dataset.
    '''
    dataloader = dict()
    dataset = dict()
    transformation = transforms.Compose(
        [   transforms.ToTensor()
         
            #transforms.Normalize(
            #    [0.5 for _ in range(CHANNELS_IMG)],
            #    [0.5 for _ in range(CHANNELS_IMG)],
            #),
            #0.5 for white/black. Random normalized values for colored.
            
        ]
    )
    dataset = CustomImageDataset(KERAS_PATH, data_type, transform = transformation)
    # DataLoader with pin_memory and num_workers for faster data loading
    #DataLoader requires the passed param to be preprocessed as img class w/
    #__getitem__ and __len__.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,\
                                 shuffle=True, num_workers=4)
    return dataloader, dataset



transformation = transforms.Compose(
        [   transforms.ToTensor(),
            #transforms.RandomHorizontalFlip(p=0.5)
            #transforms.Normalize(
            #    [0.5 for _ in range(CHANNELS_IMG)],
            #    [0.5 for _ in range(CHANNELS_IMG)],
            #),
            #0.5 for white/black. Random normalized values for colored.
        ]
    )
# only if my brain comes back

#Debug

a = CustomImageDataset(KERAS_PATH, '_train/', transform=transformation)

def check_shapes(images, labels):
    for i, img in enumerate(images):
        if img.shape != (1, 256, 128):
            print(f"Image shape error at index {i}: {img.shape}")
    for i, lbl in enumerate(labels):
        if lbl.shape != (6, 256, 128):
            print(f"Label shape error at index {i}: {lbl.shape}")

def check_shape(images, labels):
    prev_i = images[0]
    for i, img in enumerate(images):
        if prev_i.shape != img.shape:
            print(f'{i}th im shape {img.shape} from {prev_i.shape}') 
        prev_i = img
    prev_l = labels[0]
    for i, lbl in enumerate(labels):
        if prev_l.shape != lbl.shape:
            print(f'{i}th lbl shape {lbl.shape} from {prev_l.shape}') 
        prev_i = img

#check_shape(a[0], a[1])
#CustomImageDataset return tuples of (im, lbl)
#a[0] -> ims a[1] -> lbls
print(a[0])
#print(a[0][0], a[0][1], a[0][2])
#print(a[1][0], a[1][1], a[1][2])

#b = DataLoader(a, 12, shuffle = True)
#for images, labels in b:
#    print(images.shape)
#    print(labels.shape)