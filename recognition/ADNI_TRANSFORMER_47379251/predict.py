'''showing example usage of your trained model. 
Print out any results and / or provide visualisations where applicable'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from modules import *
from train import *
from dataset import *
from utils import *

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if epoch%10==0: progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        # state = {"model": net.state_dict(),
        #       "optimizer": optimizer.state_dict(),
        #       "scaler": scaler.state_dict()}
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    # os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    # with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
    #     appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

# if usewandb:
#     wandb.watch(net)
    
net.cuda()
print('Training..')
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    # list_loss.append(val_loss)
    # list_acc.append(acc)
    
    # Log training..
    # if usewandb:
    #     wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
    #     "epoch_time": time.time()-start})

    # Write out csv..
    '''with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) '''
    # print(list_loss)

print('Testing..')
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    # if usewandb:
    #     wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
    #     "epoch_time": time.time()-start})

    # Write out csv..
    '''with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) '''
    print("Test Loss:", list_loss,"Test Accuracy:", list_acc)    

# writeout wandb
# if usewandb:
#     wandb.save("wandb_{}.h5".format(args.net))
    