import os
import os.path as osp
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

"""
Contains the data loader for loading and preprocessing the ADNI dataset.
"""


#### Model hyperparameters: ####
BATCH_SIZE = 32

#### Input processing transforms: ####
# Create basic transforms for the images (using these for now, will need to add other transforms later)
BASIC_TF = transforms.Compose([transforms.ToTensor()])

#### File paths: ####
DATASET_PATH = osp.join("dataset", "AD_NC")


'''
Need to split training set data into a training and validation set.
Need to avoid data leakage ->
Need to group patient MRI image slices by the patient number, and group these
into 'bins' for each patient.
Once this is done, we can then add all slices for each patient to either the 
train or the validation set.
Should try do a stratified split of AD and NC class images.

Total of 743 AD patients and 783 NC patients in train and test sets.

Test:
AD:
- 4460 images total - 20 MRI slices per patient - 223 patients total?
- MRI slice numbers labelled differently for some patients
- 6 or 7 char patient ID numbers?
- format of image name: 'patientID_MRIslice.jpeg'
NC:
- 4540 images total - 20 MRI slices per patient - 227 patients total?

Train:
AD:
- 10400 images total - 20 MRI slices per patient - 520 patients total?
NC:
- 11120 images total - 20 MRI slices per patient - 556 patients total?

Splitting train set into a train and validation set (80/20 stratified split):
- Train: 416 AD patients and ~445 NC patients
- Validation: 104 AD patients and ~111 NC patients
'''
    
# Ignore this commented-out section for now, may just use the torchvision ImageFolder class 

# """
# Creates a custom dataset class for the ADNI dataset, loaded from images stored
# locally.
# If specified, applies transforms to the input image (data) and/or the target
# class labels (target).
# """
# class ADNI(Dataset):
#     """
#     Initialise the ADNI dataset.

#     Args:
#         data: the input image's RGB pixel intensity values (X)
#         target: target class labels (AD or NC) (y)
#         transform (torch transform): A function/transform that takes in an PIL image
#         and returns a transformed version. E.g, transforms.RandomRotation
#         target transform (torch transform): A function/transform that takes in the
#         target and transforms it.
#     """
#     def __init__(self, data, target, transform=None, target_transform=None):
#         self.data = data
#         self.target = target
#         self.transform = transform
#         self.target_transform = target_transform
    
#     """
#     Gets the total # of data points in the dataset.

#     Returns:
#         the length of the data from the input variables
#     """
#     def __len__(self):
#         return len(self.data)
    
#     """
#     Retrieve the input vars (X) and target var (y) at the given index
#     in the dataset. Apply a transform and target transform respectively to the
#     data before returning it.

#     Args:
#         index (int): the index in the dataset for which to retrieve a data point

#     Returns:
#         the X and y values for the given data point
#     """
#     def __getitem__(self, index):
#         x = self.data[index]
#         if self.transform:
#             # Apply the transform if one is parsed in the constructor
#             x = self.transform(x)
        
#         y = self.target[index]
#         if self.target_transform:
#             # Apply the target transform if one is parsed in the constructor
#             y = self.target_transform(y)
            
#         return x, y
    

"""
Load the ADNI dataset images from the given local directory/path.
Splits the data into train and test sets. Applies the specified 
transforms to these different sets.

It is assumed that the ADNI dataset images are organised in this directory
structure, relative to the project:
    - dataset_path/:
        - 'test/'
            - 'AD/
            - 'NC/'
        - 'train/'
            - 'AD/'
            - 'NC/'
By default, dataset_path is set to: './recognition/TRANSFORMER_43909856/dataset/AD_NC'.
The PyTorch ImageFolder class automatically assigns class labels for each image
based on the subfolders of either 'train' or 'test'. An image in an 'AD' dir is
assigned a class label of 'AD' (Alzheimer's Detected), and an image in an 'NC'
dir is assigned a class label of 'NC' (Normal Cognition).
            
Params:
    dataset_path (str): the directory containing the ADNI dataset images, structured
                        by the image classifications
    train_tf (torch transform): the transform to be applied to the training set data
    val_tf (torch transform): the transform to be applied to the validation set data
    test_tf (torch transform): the transform to be applied to the test set data
    batch_size (int): the number of input images to be added to each DataLoader batch

Returns:
    DataLoaders for the train and test set data
"""
def load_ADNI_data(dataset_path=DATASET_PATH, train_tf=BASIC_TF, val_tf=BASIC_TF, 
                   test_tf=BASIC_TF, batch_size=BATCH_SIZE):
    # Load the ADNI train data
    train_data = ImageFolder(root=osp.join(dataset_path, "train"), transform=train_tf)
    # Load the ADNI test data
    test_data = ImageFolder(root=osp.join(dataset_path, "test"), transform=test_tf)

    # Load the sets into DataLoader objects
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader
