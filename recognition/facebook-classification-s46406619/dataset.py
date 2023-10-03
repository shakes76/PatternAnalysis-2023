import numpy as np
import os
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

def load_data(batch_size=150):
    # set working directory
    os.chdir('C:/Area-51/2023-sem2/COMP3710/PatternAnalysis-2023/recognition/facebook-classification-s46406619')

    # import raw data files
    facebook = np.load('facebook.npz')
    edges = torch.tensor(facebook['edges'])
    features = torch.tensor(facebook['features'])
    target = torch.tensor(facebook['target'])

    # let us get some information about this dataset
    print('number of classes:', len(np.unique(target)))
    print('number of nodes:', len(features))
    print('number of directed edges:', len(edges))
    print('number of bidirectional edges:', round(len(edges) / 2))

    # train test split
    X_train, y_train, X_test, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    print('X dataset shape:', X_train.shape, X_test.shape)
    print('y dataset shape:', y_train.shape, y_test.shape)

    # convert to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # construct dataloaders
    train = data.TensorDataset(X_train, y_train)
    test = data.TensorDataset(X_test, y_test)
    trainloader = data.DataLoader(train, batch_size=batch_size, shuffle=False)
    testloader = data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloader, testloader