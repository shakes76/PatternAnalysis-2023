import torch
from dataset import test_train


model = torch.load('GCN.pt')
data = test_train()