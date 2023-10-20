""" Training module for VQVAE2 """

import time
import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_data
from modules import VQVAE, PixelSNAIL


# IO Paths
CHECKPOINT_PATH = "checkpoint/" # dir saving model snapshots
MODEL_PATH = './vqvae2.pt'          # trained model

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu when cuda is available

# Hyperparameters
INPUT_DIM = 256*256 # dimension of Input
Z_DIM = 10          # dimension of Latent Space
H_DIM = 1600        # dimension of Hidden Layer
NUM_EPOCHS = 20     # number of epoch
LR_RATE = 3e-4      # learning rate


def train_pixelsnail(args, epoch, loader, vqvae_model: VQVAE, model: PixelSNAIL, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (input, _) in enumerate(loader): # i, (input, label)
        input = input.to(device)
        vqvae_model.eval()
        model.zero_grad()

        with torch.no_grad():               # disable gradient calculation of vqvae model
            _, _, _, top, bottom = vqvae_model.encode(input)

        if args.hier == 'top':
            target = top
            out, _ = model(top)
        elif args.hier == 'bottom':
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


def train_vqvae(epoch, loader: DataLoader, model: VQVAE, optimizer, device):
    """
    Train the given VQVAE model
    
    Args:
        loader: a dataloader of training dataset
        model: the VQVAE model to train
        optimizer: the optimizer to use in training
        device: the device (cpu/gpu) to use for training
    """
    criterion = nn.MSELoss()    # loss function

    latent_loss_weight = 0.25   # weight of latent loss
    sample_size = 10            # number of samples 

    mse_sum = 0                 # avg mse = mse_sum / mse_n
    mse_n = 0

    for i, (imgs, _) in enumerate(loader):
        model.zero_grad()

        imgs = imgs.to(device)    # use gpu when available

        # Loss
        out, latent_loss = model(imgs)      # get output and corresponding latent loss
        recon_loss = criterion(out, imgs)   # get recon loss from loss function
        latent_loss = latent_loss.mean()    # get average latent loss
        loss = recon_loss + latent_loss_weight * latent_loss    # get loss
        loss.backward()                     # backward
        optimizer.step()

        mse_sum = recon_loss.item() * imgs.shape[0] # keep tracking mse
        mse_n = imgs.shape[0]

        lr = optimizer.param_groups[0]["lr"]    # learning rate

        if i % 100 == 0:                    # Give training status
            print(f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; ")
            print(f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; ")
            print(f"lr: {lr:.5f}")

            model.eval()                            # set the model in evaluation mode

            sample = imgs[:sample_size]

            with torch.no_grad():                   # disable gradient calculation
                out, _ = model(sample)              # get generated imgs

            save_image(
                torch.cat([sample, out], 0),        # concatenates sequence in dimension 0
                f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",  # file name
                nrow=sample_size,                   # number of samples
                normalize=True,                     # normalize
                range=(-1, 1),                      # range of data
            )

            model.train()                           # back to training

def main_vqvae(args):
    start_time = time.time()        # tracking execution time

    # Data
    print("Loading Data...")
    trainloader, _ = load_data(batch_size=args.batch, test=False)   # get dataloader

    # Train
    print("Training Model...")
    model = VQVAE().to(device)                                      # use gpu when available
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    # adam is good enough
    for i in range(args.epoch):
        train_vqvae(i, trainloader, model, optimizer, device)                               # train the model
        torch.save(model.state_dict(), f"{CHECKPOINT_PATH}vqvae_{str(i + 1).zfill(3)}.pt")  # save model snapshot

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))   # print execution time (loading time + training time)

def main_pixelsnail(args):
    start_time = time.time()        # tracking execution time
    
    # Data
    print("Loading Data...")
    trainloader, _ = load_data(batch_size=args.batch, test=False)    # get dataloader

    # VQVAE Model
    vqvae_model = VQVAE().to(device)                    # send to gpu if available
    vqvae_model.load_state_dict(torch.load(args.path))  # load the given model

    if args.hier == 'top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None

    # Train
    print("Training Model...")
    for i in range(args.epoch):
        train_pixelsnail(args, i, trainloader, vqvae_model, model, optimizer, scheduler, device)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'{CHECKPOINT_PATH}pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))   # print execution time (loading time + training time)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch", type=int, default=256)    # Batch size, depends on your machine
#     parser.add_argument("--epoch", type=int, default=400)   # Epoch, can be larger (>500) for a better quality
#     parser.add_argument("--lr", type=float, default=3e-4)   # learning rate

#     args = parser.parse_args()
#     print(args)

#     main_vqvae(args)

if __name__ == '__main__':
    # top
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--n_res_block', type=int, default=2)
    parser.add_argument('--n_res_channel', type=int, default=64)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--path', type=str, default=MODEL_PATH)

    args = parser.parse_args()
    print(args)

    main_pixelsnail(args)

    # bottom
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--hier', type=str, default='bottom')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--n_res_block', type=int, default=2)
    parser.add_argument('--n_res_channel', type=int, default=64)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--path', type=str, default=MODEL_PATH)

    args = parser.parse_args()
    print(args)

    main_pixelsnail(args)

