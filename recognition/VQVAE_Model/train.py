import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import structural_similarity as compute_ssim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from dataset import *
from modules import *
from tqdm import tqdm


# Hyperparameters
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_RESIDUAL_LAYER = 2
RESIDUAL_HIDDEN_DIM = 32
NUM_EMBEDDINGS = 128
EMBEDDING_DIM = 128
COMMITMENT_COST = 0.25
LEARNING_RATE = 1e-2


def train():

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create image saving path
    if not os.path.exists("./Pic"):
        os.mkdir("./Pic")

    if not os.path.exists("./Model"):
        os.mkdir("./Model")

    # Train 15000 times
    tqdm_bar = tqdm(range(15000))

    # Initialize Parameters
    i = 0
    ssim = None
    previous_loss = 1
    recon_img = None

    # Load Model, Loss function and multistep scheduler
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM,
                  COMMITMENT_COST).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = MultiStepLR(optimizer, milestones=[1000, 9000], gamma=0.1)

    model.train()

    # Define lists for plotting
    training_times_under_3000 = []
    loss_under_3000_train = []
    training_times_beyond_3000 = []
    loss_beyond_3000_train = []

    # Use tqdm bar as for loop starter
    for eq in tqdm_bar:
        train_img, _ = next(iter(training_loader))

        # Fit data and model into device
        train_img = train_img.to(device)
        model = model.to(device)

        # Fit data into model
        vq_loss, recon, perplexity, _ = model(train_img)
        loss = F.mse_loss(recon, train_img) + vq_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        i += 1

        if i <= 3000:
            # Record the loss and training epochs
            training_times_under_3000.append(i)
            loss_under_3000_train.append(loss.cpu().detach().item())
        else:
            training_times_beyond_3000.append(i)
            loss_beyond_3000_train.append(loss.cpu().detach().item())

        # Only record the smaller loss
        if loss < previous_loss:
            recon_img = recon
            trained_image = recon_img.cpu().detach().numpy()
            original_image = train_img.cpu().detach().numpy()
            ssim = compute_ssim(trained_image[0][0], original_image[0][0], data_range=2.0)

            previous_loss = loss
            print('Loss: {}'.format(loss))

            # Save generated images under the folder, all the Images have loss and ssim as their name
            loss_img = loss.cpu().detach().item()
            save_image(recon, "./Pic/No_{}_img_Loss_{}_SSIM_{}%.jpg".
                       format(i, loss_img, ssim * 100))
            # Save the generated model under the folder
            torch.save(model.state_dict(), "./Model/Vqvae.pth")

        # Show the running status every 10 epochs(show it is still working and visualisation of loss changing)
        if i % 10 == 0:
            tqdm_bar.set_description('loss: {}'.format(loss))

    # Visualize loss graph
    plt.plot(training_times_under_3000, loss_under_3000_train)
    plt.plot(training_times_beyond_3000, loss_beyond_3000_train)
    plt.show()


if __name__ == '__main__':

    train()


