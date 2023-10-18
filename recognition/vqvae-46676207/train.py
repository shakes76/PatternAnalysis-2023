""" Training module for VQVAE2 """

import time
import argparse

import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from dataset import load_data
from modules import VQVAE

# IO Paths
CHECKPOINT_PATH = "checkpoint/" # dir saving model snapshots

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu when cuda is available

# Hyperparameters
INPUT_DIM = 256*256 # dimension of Input
Z_DIM = 10          # dimension of Latent Space
H_DIM = 1600        # dimension of Hidden Layer
NUM_EPOCHS = 20     # number of epoch
LR_RATE = 3e-4      # learning rate

def train(epoch, loader: DataLoader, model: VQVAE, optimizer, device):
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

def main(args):
    start_time = time.time()        # tracking execution time

    # Data
    print("Loading Data...")
    trainloader, _ = load_data(batch_size=args.size, test=False)    # get dataloader

    model = VQVAE().to(device)                                      # use gpu when available

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    # adam is good enough

    print("Training Model...")
    for i in range(args.epoch):
        train(i, trainloader, model, optimizer, device)             # train the model

        torch.save(model.state_dict(), f"{CHECKPOINT_PATH}vqvae_{str(i + 1).zfill(3)}.pt")  # save model snapshot

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))   # print execution time (loading time + training time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=256)    # Batch size, depends on your machine
    parser.add_argument("--epoch", type=int, default=400)   # Epoch, can be larger (>500) for a better quality
    parser.add_argument("--lr", type=float, default=3e-4)   # learning rate

    args = parser.parse_args()
    print(args)

    main(args)
