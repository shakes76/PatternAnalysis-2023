import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder

class DataLoader(Dataset):
    def __init__(self, file_name):
        # extract data from the csv file
        data = pd.read_csv(file_name)

        y = data.iloc[:,-1:]
        ordinalEncoder = OrdinalEncoder()
        ordinalEncoder.fit_transform(data[['page_type']])


        x = data.iloc[:, :-1].values
        y = data.iloc[:,-1:].values

        # MIGHT WANT TO SCALE IT LOOK INTO IT want to scale it here
        x_train = x
        y_train = y

        # convert to torch tensors
        self.X_train = torch.tensor(x_train, dtype = torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return self.y_train.shape[0]
    
    def __getitem__(self, index):
        features = self.X_train[index]
        label = self.y_train[index]
        return features, label