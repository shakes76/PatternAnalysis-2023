""" OASIS MRI Dataset """

import os
from enum import Enum
from platform import node

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid

# IO Paths
match node():                                                    # root of data dir
    case 'Its-a-MacBook.local':
        DATA_PATH = '/Users/samson/Documents/UQ/COMP3710/data/keras_png_slices_data/'
    case 'Its_a_PC':
        DATA_PATH = 'D:/Documents/UQ/COMP3710/data/keras_png_slices_data/'
    case _:
        if 'vgpu' in node():
            DATA_PATH = '/home/Student/s4667620/mac_mount/data/keras_png_slices_data/'
        else:
            raise ValueError(f'Unknown hostname: {node()}. Please add your DATA_PATH in dataset.py.')
TRAIN_INPUT_PATH = DATA_PATH + 'keras_png_slices_train/'         # train input
VALID_INPUT_PATH = DATA_PATH + 'keras_png_slices_validate/'      # valid input
TEST_INPUT_PATH = DATA_PATH + 'keras_png_slices_test/'           # test input
VALID_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_validate/' # train target
TRAIN_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_train/'    # valid target
TEST_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_test/'      # test target
TRAIN_TXT = './oasis_train.txt'     # info of img for train
VALID_TXT = './oasis_valid.txt'     # info of img for valid
TEST_TXT = './oasis_test.txt'       # info of img for test

# Hyperparameters
BATCH_SIZE = 256    # Depends on your machine

class DataType(Enum):
    """ Represents types of datasets """
    TRAIN = 1   # Training set
    VALID = 2   # Validating set
    TEST = 3    # Testing set

class OASIS_MRI(Dataset):
    """ OASIS MRI Dataset """
    def __init__(self, input_folder) -> None:
        """
        Initialize a OASIS MRI Dataset
        
        Args:
            input_folder: folder of dataset
        """
        super(OASIS_MRI, self).__init__()
        
        self.input_folder = input_folder    # folder of dataset

        data_list = self.get_data_list()    # get list of data (file names with labels)

        self.inputs = []
        self.labels = []
        for line in data_list:
            img_names = line.split()        # slipt by ' ', [0]: input, [1]: target
            input = Image.open(self.input_folder + img_names[0])    # read input img
            preprocess = transforms.Compose([
                transforms.ToTensor(),      # transform to torch tensor
            ])
            input = preprocess(input)       # apply tranformation
            self.inputs.append(input)
            self.labels.append(img_names[1])

    def __getitem__(self, index):
        """
        Return a pair of data
        
        Args:
            index: index of required data

        Returns:
            input: input data
            label: label of the input data
        """
        # index will be handled by dataloader
        input = self.inputs[index]
        label = int(self.labels[index])
        return input, label
    
    def __len__(self):
        """
        Return self length
        
        Returns:
            len(self.inputs): Length of the dataset
        """
        return len(self.inputs)

    def get_data_list(self, include_label=True):
        """
        Return a list for the data

        Args:
            include_label: include label in the list or not
        
        Returns:
            file_list: list of sorted data
        """
        file_list = []
        count=0

        for file in os.listdir(self.input_folder):  # iterate through all files
            filename=os.path.splitext(file)[0]      # filename (without .png)
            filetype = os.path.splitext(file)[1]    # .png
            if include_label:
                idx = os.path.splitext(file)[0][15:len(os.path.splitext(file)[0])-4]    # take index of the image as label
                new_pair = os.path.join(filename + filetype + ' ' + idx)
            else:
                new_pair = os.path.join(filename + filetype)
            file_list.append(new_pair)
            count+=1

        file_list.sort(key=lambda item:len(str(item)), reverse=False) # sorting
        return file_list

def load_data(batch_size=BATCH_SIZE,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False) -> DataLoader :
    """
    Return a Dataloader of OASIS_MRI
    
    Args:
        batch_size: batch size
        random_seed: random seed
        shuffle: shuffle the dataset or not
        test: True: return testing dataloader
              False: return testing and validating dataloader
    
    Returns:
        test_loader: dataloader of testing dataset
        train_loader: dataloader of training dataset
        valid_loader: dataloader of validating dataset
    """

    if test:    # get the testing dataloader
        test_dataset = OASIS_MRI(       # get testing set
          TEST_INPUT_PATH,
        )

        test_loader = DataLoader(       # initialize the dataloader
            test_dataset, batch_size=batch_size, shuffle=shuffle
        )

        return test_loader

    else:       # get the training data & validating dataloader
        train_dataset = OASIS_MRI(      # get training set
            TRAIN_INPUT_PATH,
        )

        valid_dataset = OASIS_MRI(      # get validating set
            VALID_INPUT_PATH,
        )

        s_train = len(train_dataset)    # size of training set
        indices = list(range(s_train))
        split = int(np.floor(valid_size * s_train))

        if shuffle:                     # shuffle the data set
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(      # initialize the training dataloader
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=14
        )
        valid_loader = DataLoader(      # initialize the validating dataloader
            valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=14
        )

        return train_loader, valid_loader

def show_img(n_batch):
    """
    Show sample images in the OASIS_MRI dataset
    
    Args:
        n_batch: number of batches to show
    """
    dataloader = load_data(batch_size=64, test=True)    # get the dataloader

    for i, (b_x, _) in enumerate(dataloader):           # b_x: input, b_y: target
        if i < n_batch:                                       # show 3 batches of images
            imgs = make_grid(b_x)
            imgs = np.transpose(imgs,(1,2,0))
            plt.imshow(imgs)
            plt.show()
        else:
            break

if __name__ == "__main__":
    show_img(3)
