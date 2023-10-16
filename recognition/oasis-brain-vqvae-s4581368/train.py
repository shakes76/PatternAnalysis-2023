# Training module for the VQVAE
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from modules import VQVAE
from dataset import OASIS 
import matplotlib.pyplot as plt

BATCH_SIZE = 256

HIDDEN_LAYERS = 128
RESIDUAL_HIDDEN_LAYERS = 32
RESIDUAL_LAYERS = 2

EMBEDDING_DIMENSION = 64
EMBEDDINGS = 512

BETA = 0.25

LEARNING_RATE = 1e-3
EPOCHS = 20

def train(data_loader, model, optimizer, device):
    train_tilde_loss = []
    steps = 0
    for images, _ in data_loader:
        print("we are training")        
        images = images.to(device)
        optimizer.zero_grad()

        e_loss, x_tilde, x_q = model(images)
        reconstructed_loss = F.mse_loss(x_tilde, images)
        loss = e_loss + reconstructed_loss
        loss.backward()

        optimizer.step()
        train_tilde_loss.append(reconstructed_loss.item()) 
        if (steps + 1) % 100 == 0:
            print("%d iterations")
            print("reconstructed_loss: %.3f" % np.mean(train_tilde_loss[-100:]))

def test(data_loader, model, device):
    with torch.no_grad():
        loss_x, loss_q = 0., 0.
        for images, _ in data_loader:
            images = images.to(device)
            e_loss, x_tilde, x_q = model(images)
            loss_x += F.mse_loss(x_tilde, images)
            loss_q += e_loss
        loss_x /= len(data_loader)
        loss_q /= len(data_loader)

    return loss_x, loss_q

def generate_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        _, x_tilde, _ = model(images)
    return x_tilde

def main():
    now = datetime.now()
    logger = SummaryWriter("./logs/{0}".format(now.strftime("%m-%d-%Y-%H_%M_%S")))
    save_filename = './models/{0}'.format(now.strftime("%m-%d-%Y-%H_%M_%S"))

    data_path = "./keras_png_slices_data/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    oasis_data = OASIS(data_path)
    data_loader = oasis_data.data_loaders(data_path, data_path, data_path)
    fixed_images, _ = next(iter(data_loader))
    grid = make_grid(fixed_images, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    model = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    reconstructions = generate_samples(fixed_images, model, device)

    grid = make_grid(reconstructions.cpu(), nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    logger.add_image('original', grid, 0)
    best_loss = -1
    for epoch in range(EPOCHS):
        train(data_loader, model, optimizer, device)

        # replace with validation dataset
        loss, _ = test(data_loader, model, device)

        reconstruction = generate_samples(fixed_images, model, device)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        logger.add_image('reconstruction', grid, epoch + 1)
        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open("{0}/best.pt".format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)


if __name__ == '__main__':
    main()


















