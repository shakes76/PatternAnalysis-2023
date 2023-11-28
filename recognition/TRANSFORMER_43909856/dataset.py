import os
import os.path as osp
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torchdata.datapipes.iter import BucketBatcher, FileLister, Mapper, RandomSplitter, UnBatcher
from PIL import Image
from torch.utils.data.backward_compatibility import worker_init_fn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

"""
Contains the data loader for loading and preprocessing the ADNI dataset.

This resource in particular was very useful for creating custom components of the dataset
loading. Some of the code written in this file was based on the general pipeline
followed in the information on this website:
https://sebastianraschka.com/blog/2022/datapipes.html#DataPipesforDatasetsWithImagesandCSVs
"""


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
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
VAL_TF = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
# Should validation and test transforms be different? I don't see why they should be

# TODO could try some data augmentation on these transforms?
# TODO try changing from RGB images to greyscale, compare model performance


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
assigned a class label of 'AD' (0) (Alzheimer's Detected), and an image in an 'NC'
dir is assigned a class label of 'NC' (1) (Normal Cognition).
            
Params:
    dataset_path (str): the directory containing the ADNI dataset images, structured
                        by the image classifications
    tf (torch transform): the transform to be applied to the data
    batch_size (int): the number of input images to be added to each DataLoader batch
    dataset (str): "train" or "test" set

Returns:
    The given set's data
"""
def load_ADNI_data(dataset_path=DATASET_PATH, tf=TEST_TF, batch_size=BATCH_SIZE,
                   dataset="test"):
    # Load the ADNI data
    data = ImageFolder(root=osp.join(dataset_path, dataset), transform=tf)

    return data


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
Opens the PIL image specified by the given filename. Returns the opened PIL

Params:
    file_data (tuple(str, str)): a filename for the PIL image to be opened, and 
                                  label for the given data point associated with 
                                  that file ("AD" or "NC")
Returns:
    Tuple containing the opened PIL image, and the label for the given 
    data point associated with that image ("AD" or "NC")
"""
def open_image(file_data):
    filename, class_name = file_data
    return Image.open(filename).convert("RGB"), class_name


"""
Determines the class label to be assigned to a given file, based on the
contents of its filename. Returns an assignment of the class label to the filename.

Implementation assumes that the subdirs of the train dir separates datapoints of
different classes into different dirs (AD classes are in the "AD" subdir, and
NC classes are in the "NC" subdir).
Because of this, the method assumes that there must be one or more occurrences
of the particular class name ("AD" or "NC") in the given filename.

Params:
    filename (str): the file name of the given input image
Returns:
    Tuple containing the given filename, and the class for that image 
    file ("AD" - 0 or "NC" - 1)

Method throws an exception if the class label can't be determined (there are
no "AD" or "NC" substrings in the filename, indicating that the
"AD" and "NC" subdirs don't exist).
"""
def add_class_labels(filename):
    split = filename.split("AD_NC")
    if split[-1].find("AD") != -1:
        # File is in the "AD" subdir
        class_name = 0
    elif split[-1].find("NC") != -1:
        # File is in the "NC" subdir
        class_name = 1
    else:
        # If the class can't be determined, throw an exception
        return Exception(f"The class label for {split[-1]} is unknown.")
    return filename, class_name


"""
Apply a transform to images in the training set.

Params:
    image_data (tuple(PIL image, str)): contains the opened PIL image, and
                                        the class label for that image
Returns:
    The transformed input image, and the class label for that image 
    (not transformed)
"""
def apply_train_tf(image_data, train_tf=TRAIN_TF):
    image, class_name = image_data
    return train_tf(image), class_name


"""
Apply a transform to images in the validation set.

Params:
    image_data (tuple(PIL image, str)): contains the opened PIL image, and
                                        the class label for that image
Returns:
    The transformed input image, and the class label for that image 
    (not transformed)
"""
def apply_val_tf(image_data, val_tf=VAL_TF):
    image, class_name = image_data
    return val_tf(image), class_name


"""
Loads the ADNI dataset train images from the given local directory/path.
Depending on the provided train_size param, a validation set may also be
generated from data in the 'train' subdir, using a stratified split.
To prevent data leakage, the train and validation set are created using a
patient-based split. All MRI image slices for each patient are grouped 
together (per patient) - each patient is then shuffled and split into
training and validation sets. After the split is performed, the patient MRI
slices are then 'ungrouped', and data within the sets is then shuffled for each
individual image.
The method also applies the specified transforms to the train and/or validation set.

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
    Tuple with 3 values:
    The train set data, and the number of training points in the
    train set. If train_size < 1, the validation 
    set data is also returned; otherwise, a value of None is returned.
"""
def load_ADNI_data_per_patient(dataset_path=DATASET_PATH, train_tf=TRAIN_TF, 
                               val_tf=VAL_TF, batch_size=BATCH_SIZE, train_size=0.8, 
                               imgs_per_patient=N_IMGS_PER_PATIENT):
    if train_size >= 1:
        '''
        If train_size >= 1, create only a training set.
        Load the data in the same manner used to load the ADNI test set.
        '''
        train_images = load_ADNI_data(dataset_path=dataset_path, tf=train_tf,
                                           batch_size=batch_size, dataset="train")
        # Set the validation set DataLoader to none (no validation set used)
        return train_images, len(list(train_images)), None

    '''
    Create a training and validation set:
    Get all jpeg files in the train set subdirectories, then label the data 
    (with the AD or NC classes).
    '''
    AD_files = FileLister(root=osp.join(dataset_path, "train", "AD"), 
                        masks="*.jpeg", recursive=False).map(
                            add_class_labels)
    NC_files = FileLister(root=osp.join(dataset_path, "train", "NC"), 
                        masks="*.jpeg", recursive=False).map(
                            add_class_labels)
    
    '''
    Add the data into distinct batches, grouped by patient ID 
    (the batches contain the 20 MRI images per patient in the dataset).
    Performs a buffer shuffle, which shuffles the batches corresponding to each
    patient within the entire bucket (but doesn't shuffle the 20 images 
    within each patient's batch).
    '''
    AD_batch = AD_files.bucketbatch(use_in_batch_shuffle=False, 
                            batch_size=N_IMGS_PER_PATIENT, sort_key=patient_sort)
    NC_batch = NC_files.bucketbatch(use_in_batch_shuffle=False, 
                            batch_size=N_IMGS_PER_PATIENT, sort_key=patient_sort)

    '''
    Perform a stratified split of AD and NC images by the train_size argument.
    Note that the data has previously been shuffled by patient ID, within each
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
    # Get the number of training set data points:
    n_train_datapoints = len(list(train_data))


    '''
    Apply a sharding filter to the data after shuffling has taken place.
    Open the PIL images from the given dataset filenames.
    Once opened, apply the specified train and validation transforms to the images.
    '''
    train_images = train_data.sharding_filter().map(open_image).map(apply_train_tf)
    val_images = val_data.sharding_filter().map(open_image).map(apply_val_tf)

    return train_images, n_train_datapoints, val_images


"""
Plots a 4x4 grid of sample images from a specified split data set (train,
validation, or test) within the ADNI dataset.

Params:
    loader (torch DataLoader): a DataLoader for the given train, test, or validation
                               set, which contains randomly shuffled MRI image slices
    show_plot (bool): show the plot in a popup window if True; otherwise, don't
                      show the plot
    save_plot (bool): save the plot as a PNG file to the directory "plots" if
                      True; otherwise, don't save the plot
"""
def plot_data_sample(loader, show_plot=False, save_plot=False):
    ### Set-up GPU device ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU")
    else:
        print(torch.cuda.get_device_name(0))

    # Get the size of the set:
    #print(f"Data points: {len(loader.dataset)}") 

    # Plot a selection of images from a single batch of the dataset
    sample_data = next(iter(loader))
    # Create a grid of 4x4 images
    plt.figure(figsize=(4,4))
    plt.axis("off")
    # Add a title
    plt.title("Sample of ADNI dataset MRI images")
    # Plot the first 16 images in the batch
    plt.imshow(np.transpose(make_grid(sample_data[0].to(device)[:16], padding=2, 
                                      normalize=True).cpu(),(1, 2, 0)))
    
    if save_plot:
        # Create an output folder for the plot, if one doesn't already exist
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots')
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the plot in the "plots" directory
        plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', 
                        "ADNI_sample_data.png"), dpi=600)
        
    if show_plot:
        # Show the plot
        plt.show()


"""
Main method - make sure to run any methods in this file within here.
Adding this so that multiprocessing runs appropriately/correctly
on Windows devices.
"""
def main():
    pass

if __name__ == '__main__':    
    main()

