'''Containing the source code for training, validating, testing and saving your model. The model
is imported from “modules.py” and the data loader is imported from “dataset.py”.'''

from __future__ import print_function


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
from utils import *
from torchvision import models
from torchsummary import summary


# Model factory..
print('==> Building model..')
if args.net=="ViT":
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 4,
    heads = 4,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.5
)
elif args.net=="SViT":
    from vit_pytorch.vit_for_small_dataset import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 4,
    heads = 4,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net == "vit_small":
        from functools import partial
        from torch import nn
        from vit_pytorch.vit_small import VisionTransformer
        patch_size = args.patch
        img_size = size
        net = VisionTransformer(img_size=[img_size],
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=1024,
            depth=9,
            num_heads=12,
            mlp_ratio=args.vit_mlp_ratio,
            qkv_bias=True,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
elif args.net=="NaViT":
    from vit_pytorch.na_vit import NaViT
    net = NaViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
)
elif args.net=="DeepViT":
    from vit_pytorch.deepvit import DeepViT
    net = DeepViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
)
elif args.net=="CrossViT":
    from vit_pytorch.cross_vit import CrossViT
    net = CrossViT(
    image_size = size,
    num_classes = num_classes,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 64,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0.1,
    emb_dropout = 0.1
)

# For Multi-GPU
if 'cuda' in device:
    torch.cuda.empty_cache()


# Loss is CE
criterion = nn.CrossEntropyLoss()
## criterion =nn.BCELoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  
elif args.opt == "rms":
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=6e-4)    
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)      
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
sched_lr = "CosineAnnealingLR"

##### Training
scaler = torch.cuda.amp.GradScaler()
def train(epoch):
    loss_idx_value = 0
    writer = SummaryWriter()
    net.train()
    if torch.cuda.is_available(): net.cuda()
    if epoch == 0:
        print(net.train())
    train_loss = 0
    correct = 0
    total = 0
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
        if epoch%1==0: 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar("Train Loss/Epochs", train_loss, epoch) 
    if epoch==0: 
        log = "Learning Rate: " + str(args.lr) + "\nOptimizer: " + str(args.opt) + "\nModel: " + str(args.net)\
            + "\nBatch Size: " + str(args.bs) + "\nEpoch: " + str(args.n_epochs)\
            + "\nPatch Size: " + str(args.patch) + "\nDimensions: " + str(args.dimhead) + "\nConv Kernel: "\
            + "\nLR Scheduler: " +  sched_lr
        writer.add_text('Param', log, 0)
    writer.close()   
    #print(100.*correct/total)
    return train_loss


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
    total = 0
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
        if epoch%1==0: 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar("Train Loss/Epochs", train_loss, epoch) 
    valid_loss = 0.0
    acc = 0
    net.eval()     # Optional when not using Model Specific layer
    for batch_idx, (inputs, targets) in enumerate(validloader):
        # Transfer Data to GPU if available
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        valid_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (100.*correct/total) > acc: acc = 100.*correct/total
    if epoch==0: 
        log = "Learning Rate: " + str(args.lr) + "\nOptimizer: " + str(args.opt) + "\nModel: " + str(args.net)\
            + "\nBatch Size: " + str(args.bs) + "\nEpoch: " + str(args.n_epochs)\
            + "\nPatch Size: " + str(args.patch) + "\nDimensions: " + str(args.dimhead) + "\nConv Kernel: "\
            + "\nLR Scheduler: " +  sched_lr
        writer.add_text('Param', log, 0)
    writer.close()   
    #print(100.*correct/total)
    return train_loss, valid_loss, acc
    
print(net)