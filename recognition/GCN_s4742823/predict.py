'''
Simple script which loads a file of the name "Facebook_GCN.pth" and tests it.
Note that it uses the same test split as the train script. This is to ensure that the model loaded
can be accurately compared to the model trained in train.py.
'''
import torch
from dataset import load_data
from utils import device
from dataset import load_data
from train import test_model, TEST_SIZE, VAL_SIZE

if __name__ == "__main__":
    data = load_data(test_size=TEST_SIZE, val_size=VAL_SIZE)
    data = data.to(device)

    try:
        model = torch.load("Facebook_GCN.pth")
        model = model.to(device)
    except FileNotFoundError:
        print(f"Model file Facebook_GCN.pth not found.")
    else:
        test_model(model, data)