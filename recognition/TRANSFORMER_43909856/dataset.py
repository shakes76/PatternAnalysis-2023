import os
import os.path as osp
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torchdata.datapipes.iter import BucketBatcher, FileLister, Mapper, RandomSplitter, UnBatcher
from PIL import Image

"""
Contains the data loader for loading and preprocessing the ADNI dataset.
"""


#### Set-up GPU device ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")
else:
    print(torch.cuda.get_device_name(0))


#### Model hyperparameters: ####
BATCH_SIZE = 32


#### Dataset parameters: ####
# The number of MRI image slices per patient in the dataset
N_IMGS_PER_PATIENT = 20
# Dimensions to resize the original 256x240 images to (IMG_SIZE x IMG_SIZE)
IMG_SIZE = 224


#### Input processing transforms: ####
# Create basic transforms for the images (using these for now, will need to add other transforms later)
BASIC_TF = transforms.Compose([transforms.ToTensor()])
'''
Create transforms that resize the image, then crop it to create a 224x224 image.
The transforms will also normalise the RGB intensity values for the data to per-channel
means and standard deviations of 0.5 - this places intensity values in the range
[-1, 1].
'''
TRAIN_TF = transforms.Compose([
      transforms.Resize(IMG_SIZE),
      transforms.CenterCrop(IMG_SIZE),
      transforms.ToTensor(), 
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
TEST_TF = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
VAL_TF = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
# TODO should validation and test transforms be different? I don't see why they should be


#### File paths: ####
DATASET_PATH = osp.join("recognition", "TRANSFORMER_43909856", "dataset", "AD_NC")


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
- Train: 416 AD patients and ~444 NC patients (860 total)
- Validation: 104 AD patients and ~112 NC patients (216 total)
'''


"""
Loads the ADNI dataset test images from the given local directory/path. In cases
where only a train and test set are created, this method will also be used to
load the training set.
Applies the specified transforms to this set.

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
based on the subfolders in 'train' and 'test'. An image in an 'AD' dir is
assigned a class label of 'AD' (Alzheimer's Detected), and an image in an 'NC'
dir is assigned a class label of 'NC' (Normal Cognition).
            
Params:
    dataset_path (str): the directory containing the ADNI dataset images, structured
                        by the image classifications
    tf (torch transform): the transform to be applied to the data
    batch_size (int): the number of input images to be added to each DataLoader batch

Returns:
    DataLoaders for the given set's data
"""
def load_ADNI_data(dataset_path=DATASET_PATH, tf=TEST_TF, batch_size=BATCH_SIZE,
                   dataset="test"):
    # Load the ADNI data
    data = ImageFolder(root=osp.join(dataset_path, dataset), transform=tf)

    # Shuffle DataLoader data for the training set only
    if dataset == "train":
        shuffle = True
    else:
        shuffle = False
    # Load the set into DataLoader object
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)

    # Get the size of the set:
    print(f"Data points: {len(loader.dataset)}") 
    # Get the classes:
    print(f"Classes: {data.classes}")
    # Get the original file's names:
    # print(f"Images: {data.imgs}")

    # Plot a selection of images from a single batch of the dataset
    sample_data = next(iter(loader))
    # Create a grid of 8x8 images
    plt.figure(figsize=(8,8))
    plt.axis("off")
    # Add a title
    plt.title("Sample of images (test set)")
    # Plot the first 64 images in the batch
    plt.imshow(np.transpose(make_grid(sample_data[0].to(device)[:64], padding=2, normalize=True).cpu(),(1, 2, 0)))
    # Plot graph
    #plt.show()

    return loader


"""
Sort a selection of images from an input bucket based on their filename (in 
lexicographic order), so that images belonging to the same patient are grouped 
together in batches.

Implementation of this method assumes that all image filenames within the given
bucket are within the same directory locations, so that the image files can
be correctly sorted into lexicographic order. By sorting them by image file name,
the images are automatically sorted and grouped by patient ID (as the patient
ID is the first component of the image file names).

Params:
    bucket (torch object): a given 'bucket'/collection of images, with their
                           filenames included
Returns:
    A sorted version of the bucket - entries are sorted by image filename, in
    lexicographic order
"""
def patient_sort(bucket):
    return sorted(bucket)


"""
TODO

Implementation assumes that the subdirs of the train dir separates datapoints of
different classes into different dirs (AD classes are in the "AD" subdir, and
NC classes are in the "NC" subdir).
Because of this, the method assumes that there must be one or more occurrences
of the particular class name ("AD" or "NC") in the given filename.

Params:
    filename (str): the file name of the given input image
Returns:
    The given filename, and the class name for that image file ("AD" or "NC")

Method throws an exception if the class label can't be determined (there are
no "AD" or "NC" substrings in the filename, indicating that the
"AD" and "NC" subdirs don't exist).
"""
def add_class_labels(filename):
    split = filename.split("AD_NC")
    if split[-1].find("AD") != -1:
        # File is in the "AD" subdir
        class_name = "AD"
    elif split[-1].find("NC") != -1:
        # File is in the "NC" subdir
        class_name = "NC"
    else:
        # If the class can't be determined, throw an exception
        return Exception(f"The class label for {split[-1]} is unknown.")
    return filename, class_name


"""
TODO

Implementation of this method assumes that there are exactly 20 MRI image slices
per patient within the dataset. Additionally, it is assumed that there is no
data leakage between the pre-determined train and test sets (there is no patient
data within the training set, where that same patient has the same data or other
data of their own within the test set).

Params:
    dataset_path (str): the directory containing the ADNI dataset images, structured
                        by the image classifications
    train_tf (torch transform): the transform to be applied to the training set data
    val_tf (torch transform): the transform to be applied to the validation set data
    batch_size (int): the number of input images to be added to each DataLoader batch
    train_size (float): the size of data points that will be added to the
                              train set. If < 1, the remaining size will be 
                              added to a validation set 
                              (val_size = 1 - train_size).
                              Implementation assumes that this value is in the
                              range (0, 1].
    imgs_per_patient (int): the number of MRI slice images per patient which are
                            present in the dataset

Returns:
    DataLoader for the train set data. If train_size < 1, a DataLoader
    for the validation set data is also returned. Otherwise, a value of None is
    returned as well as the train set data.
"""
def load_ADNI_data_per_patient(dataset_path=DATASET_PATH, train_tf=TRAIN_TF, val_tf=VAL_TF, 
                   batch_size=BATCH_SIZE, train_size=0.8, imgs_per_patient=N_IMGS_PER_PATIENT):
    if train_size == 1:
        '''
        If train_size == 1, create only a training set.
        Load the data in the same manner used to load the ADNI test set.
        '''
        train_loader = load_ADNI_data(dataset_path=dataset_path, tf=train_tf,
                                           batch_size=batch_size)
        # Set the validation set DataLoader to none (no validation set used)
        return train_loader, None

    # Create a training and validation set:
    # Get all jpeg files in the train set subdirectories; label data (AD and NC classes)
    AD_files = FileLister(root=osp.join(dataset_path, "train", "AD"), 
                        masks="*.jpeg", recursive=False).map(add_class_labels)
    NC_files = FileLister(root=osp.join(dataset_path, "train", "NC"), 
                        masks="*.jpeg", recursive=False).map(add_class_labels)
    
    '''
    Add the data into distinct batches, grouped by patient ID 
    (the batches contain the 20 MRI images per patient in the dataset).
    Performs a buffer shuffle, which shuffles the batches corresponding to each
    patient within the entire bucket (but doesn't shuffle the 20 images 
    within each patient's batch).
    '''
    AD_batch = BucketBatcher(AD_files, use_in_batch_shuffle=False, 
                            batch_size=N_IMGS_PER_PATIENT, sort_key=patient_sort)
    NC_batch = BucketBatcher(NC_files, use_in_batch_shuffle=False, 
                            batch_size=N_IMGS_PER_PATIENT, sort_key=patient_sort)

    '''
    Perform a stratified split of AD and NC data into a train and valdation
    set. The split of the data is determined by the train_size argument.
    Note that the data has previously been shuffled by patient ID within each
    of the two classes.
    '''
    val_size = 1 - train_size
    AD_train, AD_val = AD_batch.random_split(weights={"train": train_size,
                                                      "validation": val_size},
                                                      total_length=len(list(AD_batch)),
                                                      seed=2)
    NC_train, NC_val = NC_batch.random_split(weights={"train": train_size,
                                                    "validation": val_size},
                                                    total_length=len(list(NC_batch)),
                                                    seed=3)
    '''
    Combine the AD and NC class splits into combined train and validation sets.
    Once combined, unbatch the data (so that data images are no longer batched
    by patient). 
    Then, shuffle all data images so that the entirety of a patient's
    data is not placed together (in one particular section of the dataset).
    '''
    train_data = AD_train.concat(NC_train).unbatch().shuffle()
    val_data = AD_val.concat(NC_val).unbatch().shuffle()

    # Set up the training and validation set DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    # TODO convert iterables into maps - this may make the dataloaders in this
    # method perform more similarly to the other dataloaders

    return train_loader, val_loader
    
