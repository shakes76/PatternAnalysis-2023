'''Containing the source code for training, validating, testing and saving your model. The model
is imported from “modules.py” and the data loader is imported from “dataset.py”.'''

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
from dataset import *
from utils import *

## Make sure to plot the losses and metrics during training

# Model factory..
print('==> Building model..')
# if args.net=="vit_small":
#     from models.vit_small import ViT
#     net = ViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 2,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# elif args.net=="vit_tiny":
#     from models.vit_small import ViT
#     net = ViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 2,
#     dim = int(args.dimhead),
#     depth = 4,
#     heads = 6,
#     mlp_dim = 256,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# elif args.net=="simplevit":
#     from models.simplevit import SimpleViT
#     net = SimpleViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 2,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512
# )
# elif args.net=="vit":
#     net = ViT(
#     image_size = size,
#     patch_size = args.patch,
#     num_classes = 2,
#     dim = int(args.dimhead),
#     depth = 6,
#     heads = 8,
#     mlp_dim = 3072,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )
# elif args.net=="vit_timm":
#     import timm
#     net = timm.create_model("vit_base_patch16_384", pretrained=True)
#     net.head = nn.Linear(net.head.in_features, 10)

if args.net=="vit":
    # net = ViT(
    # image_size = size,
    # patch_size = args.patch,
    # num_classes = 2,
    # dim = int(args.dimhead),
    # depth = 6,
    # heads = 8,
    # mlp_dim = 3072,
    # dropout = 0.1,
    # emb_dropout = 0.1
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 2,
    dim = int(args.dimhead),
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    torch.cuda.empty_cache()
    # net = torch.nn.DataParallel(net) # make parallel
    # cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)