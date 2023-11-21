import numpy as np
import numpy.random as random
import torch
import sys
import os

random.seed(42) # set random seed for reproducibility
os.chdir(sys.path[0]) # set working directory

class Data:
    def __init__(self, X, y, edges, train_split, test_split):
        self.X = X
        self.y = y
        self.edges = edges
        self.train_split = train_split # train split
        self.test_split = test_split # test split

def load_data(quiet=False, train_split=None, test_split=None, test_size=0.2):
    # import data and convert to tensors
    facebook = np.load('facebook.npz')
    edges = torch.transpose(torch.tensor(facebook['edges']), 0, 1)
    X = torch.tensor(facebook['features'])
    y = torch.tensor(facebook['target'])

    # let us get some information about this dataset
    if not quiet:
        print('number of classes:', len(np.unique(y)))
        print('features shape:', X.shape)
        print('edges shape:', edges.shape)

    # create train test split, stored as indices.
    if train_split is None:
        train_split = []
        test_split = []
        split = random.random(size=int(np.round(len(y))))
        for i in range(len(split)):
            if split[i] < test_size:
                test_split.append(i)
            else:
                train_split.append(i)

    return Data(X, y, edges, train_split, test_split)