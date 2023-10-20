from dataset import *
import torch.optim as optim


def train_model():
    # Set processing to GPU
    if torch.cuda.is_available():
        print("Using GPU.")
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    
    train_loader, transform = load_data_celeba()

    # Making generator
    netG = StyleGANGenerator(z_dim, init_channels, init_resolution).to(device)

    # Making discriminator
    netD = Discriminator(1).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    return
