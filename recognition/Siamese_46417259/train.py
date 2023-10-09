import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms.v2 as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from modules import SiameseTwin, SiameseNeuralNet, SiameseMLP
from dataset import PairedDataset, load_data

TRAIN_PATH = '/home/groups/comp3710/ADNI/AD_NC/train/'
TEST_PATH = '/home/groups/comp3710/ADNI/AD_NC/test/'
RESULTS_PATH = "/home/Student/s4641725/COMP3710/project_results/"

# Loss Functions and Optimizers -----------------------------------
def contrastive_loss(x1:torch.Tensor,x2:torch.Tensor, sameness:torch.Tensor, margin:float=1.0):
    
    difference = nn.PairwiseDistance(x1, x2)
    loss = (sameness * torch.pow(difference, 2) + 
            (1 - sameness) * torch.max(0, margin - torch.pow(difference, 2)))
    loss = torch.mean(loss)
    return loss


# def weights_initialisation(model:nn.Module):
#     classname = model.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.01)
#         nn.init.normal_(model.bias.data)
#     elif classname.find('Linear') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.2)
        

# Training Loop ----------------------------------
def initialise_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)

    siamese_net = SiameseNeuralNet().to(device)
    print(siamese_net)

    criterion = contrastive_loss
    optimiser = optim.Adam(siamese_net.parameters(), lr=1e-3, betas=(0.9, 0.999))

    return siamese_net, criterion, optimiser

def load_from_checkpoint(filename:str):
    pass

def save_checkpoint():
    pass

def train_and_eval():
    save_checkpoint()



# new code here

if __name__ == "__main__":
    initialise_training()
    # load_from_checkpoint()

    