import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, features, edges, target):
        self.X = features
        self.edges = edges
        self.y = target

def load_data(batch_size=150):
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

    # train test split
    #X_train, y_train, X_test, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    return Data(features, edges, target)