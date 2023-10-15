import time

import torch
import torch.nn as nn
from torchvision.utils import save_image
import argparse

from dataset import *
from modules import *
from utils import *

# IO Paths
CHECKPOINT_PATH = "checkpoint/"

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
INPUT_DIM = 256*256 # dimension of Input
Z_DIM = 10          # dimension of Latent Space
H_DIM = 1600        # dimension of Hidden Layer
NUM_EPOCHS = 20     # number of epoch
LR_RATE = 3e-4      # learning rate

def train(epoch, loader: DataLoader, model: VQVAE, optimizer, scheduler, device):
    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 10

    mse_sum = 0
    mse_n = 0

    for i, (img, _) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum = recon_loss.item() * img.shape[0]
        mse_n = img.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        if i % 100 == 0:
            # Give training status
            print(f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; ")
            print(f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; ")
            print(f"lr: {lr:.5f}")

            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            save_image(
                torch.cat([sample, out], 0),
                f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()

def main(args):
    start_time = time.time()

    # Data
    print("Loading Data...")
    trainloader, _ = load_data(batch_size=args.size, test=False)

    model = VQVAE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    print("Training Model...")
    for i in range(args.epoch):
        train(i, trainloader, model, optimizer, scheduler, device)

        torch.save(model.state_dict(), f"{CHECKPOINT_PATH}vqvae_{str(i + 1).zfill(3)}.pt")

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=256)    # Depends on your machine
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()

    print(args)

    main(args)
