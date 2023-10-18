"""
Imports Here
"""
from dataset import trainloader
from dataset import testloader
from dataset import trainaccloader
from dataset import trainshape
from dataset import testshape

from model import VisionTransformer
from model import Attention
from model import TransBlock

import time
import torch
import torch.nn as nn
import torch.optim as optim

"""for results"""
TRAIN_LOSS = []
TRAIN_ACC = []

"""
function to train the model
"""
def train(model, dataloader, accloader, lossfunc, optimiser, lr=0.1, momentum=0.9, batchsize=16, nepochs=10):
    device = next(model.parameters()).device # check what device the net parameters are on
    
    """training"""
    for i in range(nepochs): # for each epoch
        epoch_loss = 0
        model.train()
        n_batches = 0
        time1 = time.time()
        for (x, y) in dataloader: # for each mini-batch
            optimiser.zero_grad()
            loss = lossfunc(model.forward(x), y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss
            n_batches += 1
        time2 = time.time()
        print("Done an epoch", time2-time1)
        epoch_loss /= n_batches
    
        """evaluating"""
        model.eval()
        accuracy = test(model, accloader)

        """get performance"""
        TRAIN_LOSS.append(epoch_loss.item())
        TRAIN_ACC.append(accuracy)

"""
function to test the model
"""
def test(model, dataloader):
    with torch.no_grad(): # disable automatic gradient computation for efficiency
        device = next(model.parameters()).device
        
        """make predictions"""
        pcls = []
        items = 0
        time1=time.time()
        for x, y in dataloader:
            x = x.to(device)
            pcls.append(abs(y.cpu()-torch.max(model(x), 1)[1].cpu()))
            items += 1
        time2 = time.time()
        print("found accuracy in:", time2-time1)

        """get accuracy"""
        pcls = torch.cat(pcls) # concat predictions on the mini-batches
        accuracy = 1 - (pcls.sum().float() / items)
        print("accuracy:", accuracy)
        return accuracy
    
"""model training"""
batchsize=16
N, Np, P = trainshape()
model = VisionTransformer(inputsize=(batchsize, Np, P), embed=128, fflscale=2, nblocks=4)
criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=1e-4)
start = time.time()
train(model, trainloader(batchsize=batchsize), trainaccloader(), criterion, optimiser, nepochs=10)
end = time.time()
print("training time: ", end-start)
test(model, testloader())
print(TRAIN_LOSS)
print(TRAIN_ACC)
