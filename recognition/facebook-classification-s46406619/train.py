import torch
from modules import *
from dataset import *

def run_training(lr=0.02, num_epoch=100):
    data = load_data()
    model = GCN()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion=torch.nn.CrossEntropyLoss()

    model.train(data, criterion, optimizer, num_epoch)

run_training()