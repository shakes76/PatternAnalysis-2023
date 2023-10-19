# Training module for the VQVAE
import os
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from modules import VQVAE, PixelCNN, Decoder
from dataset import data_loaders, see_data
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

BATCH_SIZE = 256

HIDDEN_LAYERS = 128
RESIDUAL_HIDDEN_LAYERS = 32
RESIDUAL_LAYERS = 2

EMBEDDING_DIMENSION = 128
EMBEDDINGS = 128

BETA = 0.25

LEARNING_RATE = 1e-3
EPOCHS = 3
torch.cuda.empty_cache()

def ssim_batch(x, x_tilde):
    ssims = []
    for i in range(x.shape[0]):
        calculated_ssim = ssim(x[i, 0], x_tilde[i, 0], data_range=(x_tilde[i, 0].max() - x_tilde[i, 0].min()))
        ssims.append(calculated_ssim)
    return ssims


def calculate_batch_ssim(data_loader, model, device):
    with torch.no_grad():
        images = next(iter(data_loader))
        images = images.to(device)
        _, x_tilde, _ = model(images)
        x_tilde = x_tilde.cpu().detach().numpy()
        images = images.cpu().detach().numpy()
        calculated_ssims = ssim_batch(images, x_tilde)
        avg_ssim = sum(calculated_ssims)/len(calculated_ssims)
    return avg_ssim
        

def test(data_loader, model, device):
    with torch.no_grad():
        loss_x, loss_q = 0., 0.
        for images in data_loader:
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
    time_rep = now.strftime("%m-%d-%Y-%H_%M_%S")
    logger = SummaryWriter("./logs/{0}".format(time_rep))
    save_filename = "./models/{0}".format(time_rep)
    os.makedirs(save_filename, exist_ok=True)
    train_path = "./keras_png_slices_data/keras_png_slices_train"
    test_path= "./keras_png_slices_data/keras_png_slices_test"
    val_path = "./keras_png_slices_data/keras_png_slices_validate"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, test_loader, val_loader = data_loaders(train_path, test_path, val_path)
    fixed_images = next(iter(train_loader))
    grid = make_grid(fixed_images, nrow=8)
#    plt.imshow(grid.permute(1, 2, 0))
#    plt.show()
    model = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)
    if True:
        model.load_state_dict(torch.load("./models/10-20-2023-09_14_42/best.pt"))
        pixel_cnn = train_pixelcnn(model, train_loader, device, save_filename)
        
        vqvae_decoder = Decoder(EMBEDDING_DIMENSION, HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS)

        generate_novel_brains(pixel_cnn, vqvae_decoder, test_loader, device, logger)
        return
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    reconstructions = generate_samples(fixed_images, model, device)

    grid = make_grid(reconstructions.cpu(), nrow=8)

#    plt.imshow(grid.permute(1, 2, 0))
#    plt.show()
    logger.add_image('original', grid, 0)
    best_loss = -1
    train_tilde_loss = []
    avg_ssims = []
    avg_ssims.append(0)
    steps = 0
    for epoch in range(EPOCHS):
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            e_loss, x_tilde, x_q = model(images)
            reconstructed_loss = F.mse_loss(x_tilde, images)
            loss = e_loss + reconstructed_loss
            loss.backward()

            optimizer.step()
            train_tilde_loss.append(reconstructed_loss.item()) 

        print("Reconstruction Loss: %.3f" % np.mean(train_tilde_loss[-100:]))

        loss, _ = test(val_loader, model, device)
        batch_avg_ssim = calculate_batch_ssim(val_loader, model, device)
        avg_ssims.append(batch_avg_ssim)
        reconstruction = generate_samples(fixed_images, model, device)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        logger.add_image('reconstruction', grid, epoch + 1)
        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            print("EPOCH{0}\n".format(epoch + 1))
            with open("{0}/best.pt".format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)

    plt.figure()
    plt.plot(train_tilde_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("./logs/reconstruction_error_training.png")
    plt.figure()
    plt.plot(avg_ssims)
    plt.xlabel("EPOCH")
    plt.ylabel("SSIM")
    plt.savefig("./logs/training_ssims.png")
   
    with open('{0}/model_{1}.pt'.format(save_filename, 'final'), 'wb') as f:
        torch.save(model.state_dict(), f)

    model.load_state_dict(torch.load("./models/10-20-2023-08_06_56/best.pt"))
    model.to(device)
    model.eval()

    pixel_cnn = train_pixelcnn(model, train_loader, device, save_filename)
        
    vqvae_decoder = Decoder(EMBEDDING_DIMENSION, HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS)

    generate_novel_brains(pixel_cnn, vqvae_decoder, test_loader, device, logger)

    
        #new_model = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)
    #new_model.load_state_dict(torch.load("./models/10-17-2023-23_34_52/best.pt"))
    #new_model.to(device)
    #new_model.eval()
    #test_trained_model(test_loader, new_model, device)

def train_pixelcnn(vqvae_model: VQVAE, train_loader, device, save_path):
    cnn_model = PixelCNN(1, 128, 1)
    cnn_model = cnn_model.to(device)
    vqvae_model = vqvae_model.to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    loss = 0
    best_loss = -1
    for epoch in range(EPOCHS):
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            _, _, x_q = vqvae_model(images)

            cnn_output = cnn_model(images)
            print(cnn_output.shape, x_q.shape)

            loss = F.mse_loss(cnn_output, x_q)

            loss.backward()
            optimizer.step()
            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
            
        print(f"EPOCH: {epoch}, LOSS: {loss}, BEST_LOSS: {best_loss}")

    with open(f"{save_path}/model_cnn_final.pt", 'wb') as f:
        torch.save(cnn_model.state_dict(), f)

    return cnn_model
    
def generate_novel_brains(cnn_model, vqvae_decoder, test_loader, device, logger):
    test_images = next(iter(test_loader))
    test_images.to(device)

    cnn_quantized = cnn_model(test_images)

    x_reconstructed = vqvae_decoder(cnn_quantized)

    grid = make_grid(x_reconstructed.cpu(), nrow=8, range=(-1, 1), normalize=True)
    logger.add_image('cnn_reconstruction', grid, 0)

    






if __name__ == '__main__':
    main()





