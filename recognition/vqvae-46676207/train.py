from utils import *
from dataset import *
from modules import *

import argparse
import os
import sys

def train_vae(trainloader, n_epochs, model, optimizer, criterion):
    """
    Train the VAE model.
    """
    # Start training
    for epoch in range(n_epochs):
        loop = tqdm(enumerate(trainloader))
        for i, (x, y) in loop:
            # Forward pass
            x = x.to(device).view(-1, INPUT_DIM) # -1: auto decide
            x_reconst, mu, sigma = model(x)

            # loss, formulas
            reconst_loss = criterion(x_reconst, x)                                          # Reconstruction Loss
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))    # Kullback-Leibler Divergence
            loss = reconst_loss + kl_div

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, n_epochs, loss.item()))

    return model

def train_vqvae(epoch, loader: DataLoader, model, optimizer, scheduler, device):
    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
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

        #if dist.is_primary():
        lr = optimizer.param_groups[0]["lr"]

        # loader.set_description(
        #     (
        #         f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
        #         f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
        #         f"lr: {lr:.5f}"
        #     )
        # )

        if i % 100 == 0:
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


def main_vae():
    """
    train & save the VAE model
    """
    start_time = time.time()
    print("Program Starts")
    print("Device:", device)

    # Data
    print("Loading Data...")
    trainloader, validloader = load_data(test=False)
    
    # Model, LossFunction, Optmizer
    model = VAE(INPUT_DIM, Z_DIM, H_DIM).to(device)
    criterion = nn.BCELoss(reduction="sum") # loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    # Train
    print("Training Model...")
    train_start_time = time.time()
    model = train_vae(trainloader, NUM_EPOCHS, model, optimizer, criterion)
    print("Training Time: %.2f min" % ((time.time() - train_start_time) / 60))
    torch.save(model.state_dict(), MODEL_PATH)

    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))


def main_vqvae(args):
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Data
    print("Loading Data...")
    trainloader, validloader = load_data(test=False)

    model = VQVAE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = torch.CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(trainloader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    print("Training Model...")
    for i in range(args.epoch):
        train_vqvae(i, trainloader, model, optimizer, scheduler, device)

        # if dist.is_primary():
        torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    # main_vae()
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)

    args = parser.parse_args()

    print(args)

    main_vqvae(args)

