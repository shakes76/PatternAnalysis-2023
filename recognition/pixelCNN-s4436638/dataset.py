import torch
from torch.utils.data import Dataset
import glob
import numpy as np

class GetADNITrain(Dataset):
    def __init__(self, path, train_split=0.9):
