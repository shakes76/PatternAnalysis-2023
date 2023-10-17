"""
Created on Wednesday October 18 

This script is intended for managing the retrieval of the OASIS Brain dataset,
which consists of 9,000 images that have been downloaded. 
It involves establishing a data loader and undertaking essential preprocessing steps before commencing the training.
The dataset provided by the COMP3710 course staff is a preprocessed version derived from the original OASIS Brain dataset.
@author: Aniket Gupta 
@ID: s4824063

"""

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample