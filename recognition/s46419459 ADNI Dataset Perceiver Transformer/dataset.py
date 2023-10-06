import torch
from torch.utils.data import Dataset

import numpy as np

path = "C:\Users\dcp\Documents\OFFLINE-Projects\DATASETS\ADNI"
# path = ...

class ADNIData(Dataset):

    def __init__(self, csv_file, root_dir, transform = None) -> None:
        pass