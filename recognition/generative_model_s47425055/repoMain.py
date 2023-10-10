# Tara Bashirzadeh 

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from skimage.metrics import structural_similarity as ssim

# Constants
BATCH_SIZE = 32
N_EPOCHS = 60
PRINT_INTERVAL = 100
DATASET_PATH = './OASIS'
NUM_WORKERS = 1
INPUT_DIM = 1
DIM = 256
K = 512
LAMDA = 1
LR = 1e-3
DEVICE = torch.device('cuda')

# Directories
Path('models').mkdir(exist_ok=True)
Path('samples').mkdir(exist_ok=True)

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


# function to compute SSIM
def compute_ssim(x, x_tilde):
    # Ensure that the tensors are detached and moved to the CPU
    x_np = x.cpu().detach().numpy()
    x_tilde_np = x_tilde.cpu().detach().numpy()
    
    # Get batch size
    batch_size = x_np.shape[0]
    
    # Initialize a list to store SSIM values for each image in the batch
    ssim_values = []

    # Calculate SSIM for each image in the batch
    for i in range(batch_size):
        ssim_val = ssim(x_np[i, 0], x_tilde_np[i, 0], data_range=1)  # Assuming the images are (batch, channel, height, width), and channel=1 for grayscale
        ssim_values.append(ssim_val)
    
    # Calculate mean SSIM for the batch
    mean_ssim = np.mean(ssim_values)
    return mean_ssim

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # z_e_x - (B, D, H, W)
        # emb   - (K, D)

        emb = self.embedding.weight
        z_e_x_reshaped = z_e_x.permute(0, 2, 3, 1).contiguous().view(-1, z_e_x.shape[1])  # (B*H*W, D)
        emb_reshaped = emb  # since emb is already (K, D)

        # Calculate distances between reshaped tensors
        z_e_x_norm = (z_e_x_reshaped**2).sum(1, keepdim=True)  # (B*H*W, 1)
        emb_norm = (emb_reshaped**2).sum(1, keepdim=True).t()  # (1, K)

        dists = z_e_x_norm + emb_norm - 2 * torch.mm(z_e_x_reshaped, emb_reshaped.t())  # (B*H*W, K)

        dists = dists.view(z_e_x.shape[0], z_e_x.shape[2], z_e_x.shape[3], -1)  # reshape back to (B, H, W, K)
        latents = dists.min(-1)[1]
        return latents


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents, z_e_x

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        latents, z_e_x = self.encode(x)
        x_tilde, z_q_x = self.decode(latents)
        return x_tilde, z_e_x, z_q_x


# Data preprocessing
preproc_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

# Data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/train_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/validat_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/test_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# Model setup
model = VectorQuantizedVAE(INPUT_DIM, DIM, K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)  # for example, reduce LR every 10 epochs by 10%

def train():
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time()
        x = x.to(DEVICE)

        opt.zero_grad()

        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad()

        loss_recons = F.mse_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True)

        # Straight-through estimator
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        # Vector quantization objective
        model.codebook.zero_grad()
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = LAMDA * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward()
        opt.step()

        N = x.numel()
        nll = Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
        log_px = nll.sum() / N + np.log(128) - np.log(K * 2)
        log_px /= np.log(2)

        train_losses.append((log_px.item(), loss_recons.item(), loss_vq.item()))

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.mean(train_losses[-PRINT_INTERVAL:], axis=0),
                time.time() - start_time
            ))
        #return np.asarray(train_loss).mean(0)[0]  # return average reconstruction loss

def validate():
    model.eval()  # Switch to evaluation mode
    val_loss = []
    ssim_accum = 0.0  # Accumulator for SSIM scores
    batch_count = 0   # Counter for batches
    with torch.no_grad():  # No gradient required for validation
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(DEVICE)
            x_tilde, z_e_x, z_q_x = model(x)
            loss_recons = F.mse_loss(x_tilde, x)
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            val_loss.append(to_scalar([loss_recons, loss_vq]))

            # Compute SSIM for the current batch and accumulate
            ssim_accum += compute_ssim(x, x_tilde)
            batch_count += 1
    # Calculate the average SSIM for all batches
    avg_ssim = ssim_accum / batch_count

    # Display the average validation loss
    mean_val_loss = np.asarray(val_loss).mean(0)
    print('\nValidation Loss: {}'.format(mean_val_loss))
    # print average ssim score
    print(f"Average Validation SSIM: {avg_ssim:.4f}")

    # Append the metrics to the lists
    val_losses.append(mean_val_loss)
    ssim_scores.append(avg_ssim)
    
    return np.asarray(val_loss).mean(0), avg_ssim  # return SSIM score and loss

def generate_samples():
    model.eval()  # make sure model is in eval mode
    x, _ = next(iter(test_loader)) # x, _ = test_loader.__iter__().next()
    x = x[:32].to(DEVICE)
    x_tilde, _, _ = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2

    dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
    save_image(
        images,
        f'samples/vqvae_reconstructions_{dataset_name}.png',
        nrow=8
    )

    # Display the images
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

# Constants for determining the importance of SSIM and reconstruction loss
ALPHA = 0.5  # weight for SSIM, range [0, 1]
BETA = 1 - ALPHA  # weight for reconstruction loss

BEST_METRIC = -999  # initial value for the combination metric
BEST_SSIM = 0  # just for logging purposes
BEST_RECONS_LOSS = 999  # just for logging purposes

save_interval = 10

train_losses = []
val_losses = []
ssim_scores = []

for epoch in range(1, N_EPOCHS):
    print(f"Epoch {epoch}:")
    train()
    
    # Modify this line to unpack both loss and SSIM
    val_loss, val_ssim = validate()
    # Calculate the combined metric
    combined_metric = ALPHA * val_ssim + BETA * (1 - val_loss[0])  # assuming lower reconstruction loss is better, thus the (1 - val_loss[0])

    # Check the combined metric for improvements
    if combined_metric > BEST_METRIC:                        
        BEST_METRIC = combined_metric
        BEST_SSIM = val_ssim
        BEST_RECONS_LOSS = val_loss[0]
        print("Saving model based on improved combined metric!")
        dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
        # Save model and generate samples every 10 epochs
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f'samples/checkpoint_epoch{epoch}_vqvae.pt') 
    else:
        print(f"Not saving model! Last best combined metric: {BEST_METRIC:.4f}, SSIM: {BEST_SSIM:.4f}, Reconstruction Loss: {BEST_RECONS_LOSS:.4f}")

    # Generate samples at the end of each epoch
    generate_samples()

    # Step the scheduler to adjust learning rate
    scheduler.step()

plt.figure(figsize=(12, 5))

# Plot training and validation losses
plt.subplot(1, 2, 1)
train_losses_array = np.array(train_losses)
plt.plot(train_losses_array[:, 0], label="Overall Training Loss")
plt.plot(train_losses_array[:, 1], label="Reconstruction Training Loss")
plt.plot(train_losses_array[:, 2], label="VQ Training Loss")
plt.plot(np.array(val_losses)[:, 0], label="Validation Loss", linestyle="--")
plt.title("Losses over epochs")
plt.legend()

# Plot SSIM scores
plt.subplot(1, 2, 2)
plt.plot(ssim_scores, label="Validation SSIM", linestyle="--")
plt.title("SSIM over epochs")
plt.legend()

plt.tight_layout()
plt.show()