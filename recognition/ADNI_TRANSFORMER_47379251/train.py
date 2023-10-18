from __future__ import print_function
''' This file contains the source code for training, validating and testing saving ViT model. The model
is imported from “modules.py” and the data loader is imported from “dataset.py”.'''

import os
import argparse
import csv
import time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from modules import *
from dataset import *
from util import progress_bar
from torchvision import models
from torchsummary import summary

# Hyper Parameter Parser
parser = argparse.ArgumentParser(description='ADNI Training')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--net', default='CCT')
parser.add_argument('--bs', default='32')
parser.add_argument('--size', default="256")
parser.add_argument('--n_epochs', type=int, default='1000')
parser.add_argument('--patch', default='128', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
args = parser.parse_args()

# Model factory
print('==> Building model..')
if args.net=="CCT":
    from modules import CCT
    net = CCT(
        img_size = (256, 256),
        embedding_dim = 192,
        n_conv_layers = 2,
        kernel_size = 7,
        stride = 2,
        padding = 3,
        pooling_kernel_size = 3,
        pooling_stride = 2,
        pooling_padding = 1,
        num_layers = 2,
        num_heads = 6,
        mlp_ratio = 3.,
        num_classes = 2,
        positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
    )

# For Multi-GPU
if 'cuda' in device:
    torch.cuda.empty_cache()

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  
elif args.opt == "rms":
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=6e-4)    
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)      
    
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,steps_per_epoch=len(trainloader), epochs=args.n_epochs)
sched_lr = "OneCycleLR"

##### Training + Validation
scaler = torch.cuda.amp.GradScaler()
def train_valid(epoch):
    loss_idx_value = 0
    writer = SummaryWriter()
    net.train()
    if torch.cuda.is_available(): net.cuda()
    if epoch == 0:
        print(net.train())
    train_loss = 0
    correct = 0
    correct_valid = 0
    total = 0
    total_valid = 0
    print('\nEpoch: %d' % epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Tensorboard
        if epoch == 0 and batch_idx == 0:
            writer.add_graph(net, input_to_model=(inputs, targets)[0], verbose=True)
        # Write an image at every batch 0
        if batch_idx == 0:
            writer.add_image("Example input", inputs[0], global_step=epoch)
        # Train with amp
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        optimizer.zero_grad()    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        writer.add_scalar("Train Loss/Minibatches", train_loss, loss_idx_value)
        loss_idx_value += 1
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        scheduler.step()
        if epoch%1==0: 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss, 100.*correct/total, correct, total))    
    writer.add_scalar("Train Loss/Epochs", train_loss, epoch) 
    valid_loss = 0.0
    acc = 0
    net.eval()     # Set model to evaluation mode
    for batch_idx_valid, (inputs_valid, targets_valid) in enumerate(validloader):
        # Transfer Data to GPU if available
        inputs_valid, targets_valid = inputs_valid.to(device), targets_valid.to(device)
        outputs_valid = net(inputs_valid)
        loss_valid = criterion(outputs_valid, targets_valid)
        valid_loss += loss_valid.item()
        _, predicted_valid = outputs_valid.max(1)
        total_valid += targets_valid.size(0)
        correct_valid += predicted_valid.eq(targets_valid).sum().item()
        acc = 100.*correct_valid/total_valid
        if epoch%1==0: 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (valid_loss, 100.*correct_valid/total_valid, correct_valid, total_valid))    
    if epoch==0: 
        log = "Learning Rate: " + str(args.lr) + "\nOptimizer: " + str(args.opt) + "\nModel: " + str(args.net)\
            + "\nBatch Size: " + str(args.bs) + "\nEpoch: " + str(args.n_epochs)\
            + "\nPatch Size: " + str(args.patch) + "\nDimensions: " + str(args.dimhead) + "\nConv Kernel: "\
            + "\nLR Scheduler: " +  sched_lr
        writer.add_text('Param', log, 0)
    writer.close()   
    return train_loss, valid_loss, acc # return train, valid loss and valid accuracy 
    
print(net)
