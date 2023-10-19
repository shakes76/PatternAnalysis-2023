import torch
from dataset import load_data
from utils import device
from dataset import load_data
from train import test_model

if __name__ == "__main__":
    data = load_data(test_size=0, val_size=0)
    data = data.to(device)

    model = torch.load("Facebook_GCN.pth")
    model = model.to(device)

    test_model(model, data)