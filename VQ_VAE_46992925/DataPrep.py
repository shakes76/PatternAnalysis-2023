import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np  
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Torch version ", torch.__version__)


# ------------------------------------------------
# Data Loader

#path = "C:/Users/61423/COMP3710/data/keras_png_slices_data/"
path = "//puffball.labs.eait.uq.edu.au/s4699292/Documents/2023 Sem2/Comp3710/keras_png_slices_data/keras_png_slices_data/"


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

print("Loading data")

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.13242, 0.18826)
                                ])

train_data_dir = "keras_png_slices_train/"
test_data_dir = "keras_png_slices_test/"


train_dataset = ImageDataset(path+train_data_dir, transform=transform)
test_dataset = ImageDataset(path+test_data_dir, transform=transform)

# DataLoaders
# B, C, H, W
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Debugging
first_batch = 0
for batch in train_dataloader:
    first_batch = batch
    break
print("Shape of first batch is: ", first_batch.shape)
print("First batch - Mean: {} Std: {}".format(torch.mean(first_batch), torch.std(first_batch)))
plt.imshow(first_batch[0][0])
plt.title("First Training image (Normalised)")
plt.gray()
plt.show()


print("> Data Loading Finished")


# ------------------------------------------------
# Model

class VQVAE(nn.Module):
    def __init__(self, ):
        super(VQVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        
        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)        # TODO FC layer??
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=2)
        self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)
        
        # Commitment loss beta
        self.beta = 0.2
        self.alpha = 1.0
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
            
    def forward(self, x):
        # B, C, H, W
        encoded_output = self.encoder(x)
        quant_input = self.pre_quant_conv(encoded_output)
        
        # Quantisation
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
        
        # Compute pairwise distances
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        
        # Find index of nearest embedding
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))
        
        # Compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)             # TODO change to MSE
        codebook_loss = torch.mean((quant_out - quant_input.detach())**2)
        
        
        # Straight through gradient estimator
        quant_out = quant_input + (quant_out - quant_input).detach()        # Detach ~ ignored for back-prop
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        
        # Decoding
        decoder_input = self.post_quant_conv(quant_out)
        output = self.decoder(decoder_input)
        
        # Reconstruction Loss, and find the total loss
        reconstruction_loss = F.mse_loss(x, output)
        total_losses = self.alpha*reconstruction_loss + codebook_loss + self.beta*commitment_loss

        # TODO ensure the losses are balanced
        #print("The reconstruction loss makes up {}% of the total loss ({}/{})"
        #    .format(reconstruction_loss*100//(total_losses), int(reconstruction_loss), int(total_losses)))
        
        return output, total_losses
    
    
# ------------------------------------------------
# Training

########################## TODO THERE IS NO RECONSTRUCTION LOSS!!

losses = []     # for visualisation

# Hyperparams
learning_rate = 1.e-3
num_epochs = 7

model = VQVAE().to(device)
print(model)

optimiser = torch.optim.Adam(model.parameters(), learning_rate)

for epoch_num, epoch in enumerate(tqdm(range(num_epochs))):
    model.train()
    for train_batch in tqdm(train_dataloader):
        images = train_batch
        images = images.to(device, dtype=torch.float32)
        
        output, total_losses = model(images)

        optimiser.zero_grad()       # Reset gradients to zero for back-prop (not cumulative)
        total_losses.backward()     # Calculate grad
        optimiser.step()            # Adjust weights
        
    # Evaluate
    model.eval()
    
    for test_batch in tqdm(test_dataloader):
        images = test_batch

        images = images.to(device, dtype=torch.float32)         # (Set as float to ensure weights input are the same type)
        
        with torch.no_grad():
            output, total_losses = model(images)

            
    print("Epoch {} of {}. Total Loss: {}".format(epoch_num, num_epochs, total_losses))
    
    losses.append(total_losses)     # To graph losses


# -------------------------------------------------
# Visualise

# C, H, W
input_img = test_dataset[0][0]

# Reshape to B, C, H, W for the model
input_img = input_img.reshape(1, 1, input_img.size(-2), input_img.size(-1))
input_img = input_img.to(device, dtype=torch.float32)

# DEBUGGING Print the input image shape and show it.
print("Shape of the input img is: ", input_img.shape)
#plt.imshow(input_img[0][0].cpu().numpy())
#plt.gray()
#plt.show()


with torch.no_grad():  # Ensure no gradient calculation
    output, _ = model(input_img)  # Forward pass through the model

print("Shape of the output img is: ", output.shape)

# Display input and output images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_img[0][0].cpu().numpy(), cmap='gray')  # Assuming single-channel input

plt.subplot(1, 2, 2)
plt.title("Model Output")
plt.imshow(output[0][0].cpu().numpy(), cmap='gray')  # Assuming single-channel output
plt.show()

plt.plot(losses)
plt.title("Losses")
plt.xlabel("Num Epochs")
plt.ylabel("Loss")
plt.show()