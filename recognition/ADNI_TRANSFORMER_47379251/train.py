'''Containing the source code for training, validating, testing and saving your model. The model
is imported from “modules.py” and the data loader is imported from “dataset.py”.'''

from __future__ import print_function


import os
import argparse
import csv
import time
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


# Model factory..
print('==> Building model..')
if args.net=="ViT":
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.dimhead),
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
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
elif args.net=="ScalableViT":
    from vit_pytorch.scalable_vit import ScalableViT
    net = ScalableViT(
        num_classes = num_classes,
        dim = 64,                               # starting model dimension. at every stage, dimension is doubled
        heads = (2, 4, 8, 16),                  # number of attention heads at each stage
        depth = (2, 2, 20, 2),                  # number of transformer blocks at each stage
        ssa_dim_key = (40, 40, 40, 32),         # the dimension of the attention keys (and queries) for SSA. in the paper, they represented this as a scale factor on the base dimension per key (ssa_dim_key / dim_key)
        reduction_factor = (8, 4, 2, 1),        # downsampling of the key / values in SSA. in the paper, this was represented as (reduction_factor ** -2)
        window_size = (64, 32, None, None),     # window size of the IWSA at each stage. None means no windowing needed
        dropout = 0.1,                          # attention and feedforward dropout
)
elif args.net=="Dino":
    from vit_pytorch import ViT, Dino

    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    learner = Dino(
        net,
        image_size = 256,
        hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
        projection_hidden_size = 256,      # projector network hidden dimension
        projection_layers = 4,             # number of layers in projection network
        num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
        student_temp = 0.9,                # student temperature
        teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
        global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok

)
elif args.net=="CaViT":
    from vit_pytorch.na_vit import CaViT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = 1024,
    depth = 12,             # depth of transformer for patch to patch attention only
    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05    # randomly dropout 5% of the layers
)
elif args.net=="MaxViT":
    from vit_pytorch.max_vit import MaxViT
    net = MaxViT(
    num_classes = num_classes,
    dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
    dim = 96,                         # dimension of first layer, doubles every layer
    dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
    depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
    window_size = 7,                  # window size for block and grids
    mbconv_expansion_rate = 4,        # expansion rate of MBConv
    mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
    dropout = 0.1                     # dropout
)

# For Multi-GPU
if 'cuda' in device:
    print(device)
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
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)      
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
sched_lr = "CosineAnnealingLR"

##### Training
scaler = torch.cuda.amp.GradScaler()
def train(epoch):
    loss_idx_value = 0
    writer = SummaryWriter()
    net.train()
    net.cuda()
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
        if epoch%2==0: 
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar("Train Loss/Epochs", train_loss, epoch) 
    if epoch==0: 
        log = "Learning Rate: " + str(args.lr) + "\nOptimizer: " + str(args.opt) + "\nModel: " + str(args.net)\
            + "\nBatch Size: " + str(args.bs) + "\nEpoch: " + str(args.n_epochs)\
            + "\nPatch Size: " + str(args.patch) + "\nDimensions: " + str(args.dimhead) + "\nConv Kernel: " + str(args.convkernel)\
            + "\nLR Scheduler: " +  sched_lr
        writer.add_text('Param', log, 0)
    writer.close()   
    return train_loss
    
