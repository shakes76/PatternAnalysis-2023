"""
Imports Here
"""
from dataset import trainloader
from dataset import testloader
from dataset import valloader
from dataset import trainshape
from dataset import testshape

from modules import VisionTransformer
from modules import Attention
from modules import TransBlock
from modules import ConvLayer

import time
import torch
import torch.nn as nn
import torch.optim as optim

from numpy import savetxt

"""for results"""
TRAIN_LOSS = []
TRAIN_ACC = []
TRAIN_TIMES = []

"""
function to train the model
"""
def train(model, dataloader, accloader, lossfunc, optimiser, nepochs=10):    
    """training"""
    for i in range(nepochs): # for each epoch
        epoch_loss = 0
        model.train()
        n_batches = 0
        time1 = time.time()
        for (x, y) in dataloader: # for each mini-batch
            optimiser.zero_grad(set_to_none=True)
            loss = lossfunc(model.forward(x), y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.detach().item()
            n_batches += 1
        time2 = time.time()
        TRAIN_TIMES.append(round(time2-time1,3))
        epoch_loss /= n_batches

        """evaluating"""
        model.eval()
        accuracy = test(model, accloader).detach().item()

        """get performance"""
        TRAIN_LOSS.append(round(epoch_loss,5))
        TRAIN_ACC.append(round(accuracy*100,2))


"""
function to test the model
"""
def test(model, dataloader):
    with torch.no_grad(): # disable automatic gradient computation for efficiency
        """make predictions"""
        pcls = []
        items = 0
        for x, y in dataloader:
            pcls.append(abs(y.cpu()-torch.max(model(x), 1)[1].cpu()))
            items += 1

        """get accuracy"""
        pcls = torch.cat(pcls) # concat predictions on the mini-batches
        accuracy = 1 - (pcls.sum().float() / items)
        return accuracy
    
"""model training"""
batchsize=16
N, Np, L, W, H = trainshape()
model = VisionTransformer(inputsize=(batchsize, 192, 120), heads=4, embed=360, fflscale=2, nblocks=4)
criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=3e-4)
start = time.time()
train(model, trainloader(batchsize=batchsize), valloader(), criterion, optimiser, nepochs=100)
end = time.time()
print("training time: ", end-start)
print("test acc: ", test(model, testloader()))
print(TRAIN_LOSS)
print(TRAIN_ACC)
print(TRAIN_TIMES)
test(model, testloader())
print(TRAIN_LOSS)
print(TRAIN_ACC)

"""saving model"""
# model_trained = torch.jit.script(model)
# model_trained.save('model_trained.pt')
savetxt('loss.txt', TRAIN_LOSS)
savetxt('acc.txt', TRAIN_ACC)