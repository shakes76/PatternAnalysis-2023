"""Training of the VQVAE and PixelCNN"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np  
from tqdm.auto import tqdm
from PIL import Image
import torch.utils.data
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

import modules
import dataset

# Replace with preferred device and local path(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Torch version ", torch.__version__)
path = "//puffball.labs.eait.uq.edu.au/s4699292/Documents/2023 Sem2/Comp3710/keras_png_slices_data/keras_png_slices_data/"
vqvae_save_path = "//puffball.labs.eait.uq.edu.au/s4699292/Documents/2023 Sem2/Comp3710/Saved_Models/"
pixelCNN_save_path = "//puffball.labs.eait.uq.edu.au/s4699292/Documents/2023 Sem2/Comp3710/Saved_Models/"


# Hyperparameters
batch_size = 128
vqvae_num_epochs = 60
vqvae_lr = 1e-3
cnn_num_epochs = 25
cnn_lr = 1e-3

# Data (If necessary, replace with the local names of the train, validate and test folders)
print("> Loading Data")
processed_data = dataset.DataPreparer(path, "keras_png_slices_train/", "keras_png_slices_validate/", "keras_png_slices_test/", batch_size)
train_dataloader = processed_data.train_dataloader
validate_dataloader = processed_data.validate_dataloader
test_dataloader = processed_data.test_dataloader

# Models
vqvae_model = modules.VQVAE(num_embeddings=256, embedding_dim=32).to(device)
cnn_model = modules.PixelCNN(in_channels=1, hidden_channels=128, num_embeddings=256).to(device)

# Optimisers
vqvae_optimiser = torch.optim.Adam(vqvae_model.parameters(), vqvae_lr)
cnn_optimiser = torch.optim.Adam(cnn_model.parameters(), cnn_lr)

# Initialise losses (for graphing)
vqvae_training_loss = []
vqvae_validation_loss = []
cnn_training_loss = []
cnn_validation_loss = []



# --------------------------------------------------------
# VQVAE functions
# --------------------------------------------------------

def train_vqvae():

    print("> Training VQVAE")

    for epoch_num, epoch in enumerate(range(vqvae_num_epochs)):

        # Train
        vqvae_model.train()
        for train_batch in train_dataloader:
            images = train_batch.to(device, dtype=torch.float32)
            
            output, quant_loss, reconstruction_loss, _ = vqvae_model(images)
            training_loss = quant_loss + reconstruction_loss               # Can be adjusted if necessary

            vqvae_optimiser.zero_grad()         # Reset gradients to zero
            training_loss.backward()               # Calculate grad
            vqvae_optimiser.step()              # Adjust weights

        with torch.no_grad():
            vqvae_training_loss.append((quant_loss.cpu(), reconstruction_loss.cpu(), training_loss.cpu()))

        # Evaluate
        vqvae_model.eval()
        for validate_batch in (validate_dataloader):
            images = validate_batch.to(device, dtype=torch.float32)
            
            with torch.no_grad():
                output, quant_loss, reconstruction_loss, _ = vqvae_model(images)
                validation_loss = quant_loss + reconstruction_loss
                vqvae_validation_loss.append((quant_loss.cpu(), reconstruction_loss.cpu(), validation_loss.cpu()))

        print("Epoch {} of {}. Training Loss: {}, Validation Loss: {}".format(epoch_num+1, vqvae_num_epochs, training_loss, validation_loss))


def plot_vqvae_losses(show_individual_losses=False):
    # Losses are in the order (Quant, Reconstruction, Total)
    plt.title("VQVAE Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if show_individual_losses == False:
        plt.plot([loss[2] for loss in vqvae_training_loss], color='blue')
        plt.plot([loss[2] for loss in vqvae_validation_loss], color='red')
        plt.legend(["Training Loss", "Validation Loss"])
    else:
        plt.plot([loss[0] for loss in vqvae_training_loss], color='blue', ls='--')
        plt.plot([loss[0] for loss in vqvae_validation_loss], color='red', ls='--')
        plt.plot([loss[1] for loss in vqvae_training_loss], color='blue')
        plt.plot([loss[1] for loss in vqvae_validation_loss], color='red')
        plt.legend(["Training Quantisation Loss", "Validation Quantisation Loss", "Training Reconstruction Loss", "Validation Reconstruction Loss"])
    plt.show()


"""Function to test the VQVAE. Input the number of samples to show"""
def test_vqvae(num_shown=0):

    print("> Testing VQVAE")

    # Calculate losses
    vqvae_model.eval()
    test_losses = []
    for test_batch in (test_dataloader):
        images = test_batch.to(device, dtype=torch.float32)
        with torch.no_grad():
            output, quant_loss, r_loss, _ = vqvae_model(images)
            # For averaging loss
            test_losses.append((quant_loss + r_loss).cpu())
    
    print("Average loss during testing is: ", np.mean(np.array(test_losses)))

    # Show N Reconstructed Images.
    if (num_shown != 0):
        input_imgs = processed_data.test_dataset[0:num_shown]
        input_imgs = input_imgs.to(device, dtype=torch.float32)
        with torch.no_grad():
            output_imgs, _, _, encoding_indices = vqvae_model(input_imgs)

        fig, ax = plt.subplots(num_shown, 3)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0)
        ax[0, 0].set_title("Input Image")
        ax[0, 1].set_title("CodeBook Indices")
        ax[0, 2].set_title("Reconstructed Image")
        for i in range(num_shown):
            for j in range(3):
                ax[i, j].axis('off')
            ax[i, 0].imshow(input_imgs[i][0].cpu().numpy(), cmap='gray')
            ax[i, 1].imshow(encoding_indices[i].cpu().numpy(), cmap='gray')
            ax[i, 2].imshow(output_imgs[i][0].cpu().numpy(), cmap='gray')
        
        plt.show()


# Code
#train_vqvae()
#plot_vqvae_losses()
#test_vqvae(num_shown=3)
#print("> Saving Model")
#torch.save(vqvae_model, vqvae_save_path + "trained_vqvae.pth")



# --------------------------------------------------------
# PixCNN functions
# --------------------------------------------------------
vqvae_model = torch.load(vqvae_save_path + "trained_vqvae.pth")
encoder = vqvae_model.__getattr__("encoder")
quantiser = vqvae_model.__getattr__("quantiser")
decoder = vqvae_model.__getattr__("decoder")


def train_pixcnn():

    print("> Training PixelCNN")

    for epoch_num, epoch in enumerate(range(cnn_num_epochs)):
        
        cnn_model.train()

        for train_batch in train_dataloader:
            
            # Get the quantised outputs
            with torch.no_grad():
                encoder_output = encoder(train_batch.to(device))
                _, _, indices = quantiser(encoder_output)
                indices = indices.reshape(indices.size(0), 1, indices.size(1), indices.size(2)).to(device)
                            
            output = cnn_model(indices)
            
            # Compute loss
            nll = F.cross_entropy(output, indices, reduction='none')        # Negative log-likelihood
            bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))               # Bits per dimension
            training_loss = bpd.mean()

            cnn_optimiser.zero_grad()       # Reset gradients to zero for back-prop (not cumulative)
            training_loss.backward()             # Calculate grad
            cnn_optimiser.step()            # Adjust weights

        with torch.no_grad():
            cnn_training_loss.append(training_loss.cpu())
            
        cnn_model.eval()
        
        for validate_batch in validate_dataloader:
            with torch.no_grad():
                # Get the quantised outputs
                encoder_output = encoder(validate_batch.to(device))
                _, _, indices = quantiser(encoder_output)
                indices = indices.reshape(indices.size(0), 1, indices.size(1), indices.size(2)).to(device)
                output = cnn_model(indices)
            
                # Compute loss
                nll = F.cross_entropy(output, indices, reduction='none')        # Negative log-likelihood
                bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))               # Bits per dimension
                validation_loss = bpd.mean()

        with torch.no_grad():
            cnn_validation_loss.append(validation_loss.cpu())
            
        print("Epoch {} of {}. Training Loss: {}, Validation Loss: {}".format(epoch_num+1, cnn_num_epochs, training_loss, validation_loss))


def plot_cnn_loss():
    # Losses are in the order (Quant, Reconstruction, Total)
    plt.title("PixelCNN Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(cnn_training_loss, color='blue')
    plt.plot(cnn_validation_loss, color='red')
    plt.legend(["Training Loss", "Validation Loss"])
    plt.show()


def test_cnn(shown_imgs=0):
    print("> Testing PixelCNN")
    
    test_loss = []

    cnn_model.eval()

    with torch.no_grad():
        for test_batch in test_dataloader:
            # Get the quantised outputs
            encoder_output = encoder(test_batch.to(device))
            _, _, indices = quantiser(encoder_output)
            indices = indices.reshape(indices.size(0), 1, indices.size(1), indices.size(2)).to(device)
            output = cnn_model(indices)
        
            # Compute loss
            nll = F.cross_entropy(output, indices, reduction='none')        # Negative log-likelihood
            bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))               # Bits per dimension
            validation_loss = bpd.mean()
            test_loss.append(validation_loss.cpu())
    
    print("Average loss during testing is: ", np.mean(np.array(test_loss)))

    if shown_imgs != 0:
        print(" > Showing Images")

        # Inputs
        test_batch = processed_data.test_dataset[0:shown_imgs]

        encoder_output = encoder(test_batch.to(device))
        _, _, indices = quantiser(encoder_output)

        indices_shape = indices.cpu().numpy().shape

        print("Indices shape is: ", indices.cpu().numpy().shape)
        indices = indices.reshape((indices_shape[0], 1,indices_shape[1], indices_shape[2]))
        print("Indices shape is: ", indices.cpu().numpy().shape)

        # Masked Inputs (only top quarter shown)
        masked_indices = 1*indices
        masked_indices[:,:,16:,:] = -1

        gen_indices = cnn_model.sample((shown_imgs, 1, 32, 32), ind=masked_indices*1)

        fig, ax = plt.subplots(shown_imgs, 3)
        
        for a in ax.flatten():
            a.axis('off')
        
        ax[0, 0].set_title("Real")
        ax[0, 1].set_title("Masked")
        ax[0, 2].set_title("Generated")

        for i in range(shown_imgs):
            ax[i, 0].imshow(indices[i][0].long().cpu().numpy(), cmap='gray')
            ax[i, 1].imshow(masked_indices[i][0].cpu().numpy(), cmap='gray')
            ax[i, 2].imshow(gen_indices[i][0].cpu().numpy(), cmap='gray')
        plt.show()



        plt.imshow(vqvae_model.img_from_indices(gen_indices[0], (1, 32, 32, 32))[0][0].cpu().numpy(), cmap='gray')
        plt.show()

train_pixcnn()
plot_cnn_loss()
test_cnn(shown_imgs=3)
print("Saving pixel cnn")
torch.save(cnn_model, pixelCNN_save_path + "PixelCNN model.pth")