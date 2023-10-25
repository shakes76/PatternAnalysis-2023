import torch
import torch.nn as nn
from dataset import returnDataLoaders
from modules import *
import torch.optim as optim

dataloader_train, dataloader_test, dataloader_val = returnDataloaders()

losses = []
accuracies = []

def train(net, dataloader_train, dataloader_val, cross_entropy):
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    epochs = 75
    # training loop
    for epoch in range(epochs): 
        epoch_loss = 0
        net.train()
        for (x_batch, y_batch) in dataloader_train: # for each mini-batch
            optimizer.zero_grad()
            loss = cross_entropy(net.forward(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss = epoch_loss / len(dataloader_train)
        losses.append(epoch_loss)
        
        net.eval()
        acc = test(net, dataloader_val)
        print("epoch:", epoch, "accuracy:", acc, flush=True)
        accuracies.append(acc)

def test(net, dataloader_val, batch_size=16):
    with torch.no_grad():
        acc = 0
        for (x_batch, y_batch) in dataloader_val:
            acc += torch.sum((y_batch == torch.max(net(x_batch).detach(), 1)[1]), axis=0)/len(y_batch)
        acc = acc/len(dataloader_val)
    return acc

vit = VisionTransformer(input_dimen=128,
                        hiddenlayer_dimen=256,
                        number_heads=4,
                        transform_layers=4,
                        predict_num=2,
                        size_patch=(16,16))
cross_entropy = nn.CrossEntropyLoss()
train(vit, dataloader_train, dataloader_val, cross_entropy)
