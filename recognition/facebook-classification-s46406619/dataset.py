import numpy as np
import numpy.random as random
import os
import torch
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, features, edges, y, split_indices):
        self.X = features
        self.edges = edges
        self.y = y
        self.split_indices = split_indices

def load_data(test_size=0.2):
    # set working directory
    os.chdir('C:/Area-51/2023-sem2/COMP3710/PatternAnalysis-2023/recognition/facebook-classification-s46406619')

    # import raw data files
    facebook = np.load('facebook.npz')
    edges = torch.transpose(torch.tensor(facebook['edges']), 0, 1)
    features = torch.tensor(facebook['features'])
    target = torch.tensor(facebook['target'])

    # let us get some information about this dataset
    print('number of classes:', len(np.unique(target)))
    print('number of nodes:', len(features))
    print('number of directed edges:', len(edges[1]))
    print('number of bidirectional edges:', round(len(edges[1]) / 2))
    print('features shape:', features.shape)
    print('edges shape:', edges.shape)

    # create indices to determine train and test split
    split_indices = random.random(size=int(np.round(len(target))))
    for i in range(len(split_indices)):
        if split_indices[i] < test_size:
            split_indices[i] = 1 # test element
        else:
            split_indices[i] = 0 # train element

    return Data(features, edges, target, torch.tensor(split_indices))