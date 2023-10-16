'''showing example usage of your trained model. 
Print out any results and / or provide visualisations where applicable'''

from __future__ import print_function

import os
import argparse
import csv
import time
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt1

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from modules import *
from train import *
from dataset import *
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
##### Validation
def test():
    global best_acc
    # net.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
 
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    
    content = f'lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    return test_loss, acc


def test_measure():
    list_loss = []
    print('Training..')
    start = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        trainloss = train(epoch) 
        list_loss.append(trainloss)     
        scheduler.step(epoch-1) # step cosine scheduling
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")    
    # Plots
    plt.plot([i for i in range(1, args.n_epochs+1)], list_loss, label='Train')
    plt.title("Training Plot")
    plt.legend()
    plt.savefig(str(datetime.now().strftime("%H:%M:%S"))+'_Report.png')
    print('Testing..')
    net.eval()
    print(net.eval())
    start = time.time()
    val_loss, acc = test()    
    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print('END')    
    print("Test Loss:", val_loss,"Test Accuracy:", acc) 

def test_valid_measure():
    to_print = 0
    list_loss = []
    list_acc = []
    list_test=[]
    acu = []
    print('Training..')
    start = time.time()
    for epoch in range(start_epoch, args.n_epochs):
        to_print +=1
        trainloss,testloss, acc = train_valid(epoch) 
        list_loss.append(trainloss)   
        list_test.append(testloss)   
        acu.append(acc)  
        if acc >=81: break
        #scheduler.step(testloss) # REDUCELRON PLATEAU
        #scheduler.step(epoch-1) # step cosine scheduling
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")    
    print(acu)
    print('Testing..')
    net.eval()
    print(net.eval())
    start = time.time()
    val_loss, acc = test()    
    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print('END')    
    print("Test Loss:", val_loss,"Test Accuracy:", acc)
    # Plots
    plt.plot([i for i in range(1, len(list_loss)+1)], list_loss, label='Train')
    plt.plot([i for i in range(1, len(list_test)+1)], list_test, label='Valid')
    plt.title("Training Plot")
    plt.legend()
    plt.savefig(str(datetime.now().strftime("%H:%M:%S"))+'_Report.png')
    plt1.plot([i for i in range(1, len(acu)+1)], acu)
    plt1.title("Accuracy Plot")
    plt1.legend()
    plt1.savefig(str(datetime.now().strftime("%H:%M:%S"))+'_Accuracy.png')
    torch.save(net.train(), 'model_valid_stop.pth') 

test_valid_measure()      
#test_measure()           