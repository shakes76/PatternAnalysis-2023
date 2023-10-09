# Tara Bashirzadeh 

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from skimage.metrics import structural_similarity as ssim

# Constants
BATCH_SIZE = 32
N_EPOCHS = 2
PRINT_INTERVAL = 100
DATASET_PATH = '/home/groups/comp3710/OASIS'
NUM_WORKERS = 4
INPUT_DIM = 1  # 1 for grayscale
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
        dists = torch.pow(
            z_e_x.unsqueeze(1) - emb[None, :, :, None, None],
            2
        ).sum(2)

        latents = dists.min(1)[1]
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

# Data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/keras_png_slices_train', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/keras_png_slices_validate', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/keras_png_slices_test', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# Model setup
model = VectorQuantizedVAE(INPUT_DIM, DIM, K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)  # for example, reduce LR every 10 epochs by 10%

def train():
    model.train()
    train_loss = []
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

        train_loss.append(
            [log_px.item()] + to_scalar([loss_recons, loss_vq])
        )

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))

def validate():
    model.eval()  # Switch to evaluation mode
    val_loss = []
    with torch.no_grad():  # No gradient required for validation
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(DEVICE)
            x_tilde, z_e_x, z_q_x = model(x)
            loss_recons = F.mse_loss(x_tilde, x)
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            val_loss.append(to_scalar([loss_recons, loss_vq]))

    # Display the average validation loss
    print('\nValidation Loss: {}'.format(np.asarray(val_loss).mean(0)))
    
    # Compute SSIM for validation
    ssim_score = compute_ssim(x, x_tilde)
    print(f"Validation SSIM: {ssim_score}")
    
    return np.asarray(val_loss).mean(0)

def validate():
    model.eval()  # Switch to evaluation mode
    val_loss = []
    with torch.no_grad():  # No gradient required for validation
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(DEVICE)
            x_tilde, z_e_x, z_q_x = model(x)
            loss_recons = F.mse_loss(x_tilde, x)
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            val_loss.append(to_scalar([loss_recons, loss_vq]))

    # Display the average validation loss
    print('\nValidation Loss: {}'.format(np.asarray(val_loss).mean(0)))
    
    # Compute SSIM for validation
    ssim_score = compute_ssim(x, x_tilde)
    print(f"Validation SSIM: {ssim_score}")
    
    return np.asarray(val_loss).mean(0), ssim_score  # return SSIM score and loss

def generate_samples():
    model.eval()  # make sure model is in eval mode
    x, _ = test_loader.__iter__().next()
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

BEST_LOSS = 999
LAST_SAVED = -1
BEST_SSIM = 0  # initial value, assuming higher SSIM is better

for epoch in range(1, N_EPOCHS):
    print(f"Epoch {epoch}:")
    train()
    
    # Modify this line to unpack both loss and SSIM
    val_loss, val_ssim = validate()

    # Check SSIM instead of loss for performance improvements
    if val_ssim > BEST_SSIM:                        
        BEST_SSIM = val_ssim                    
        print("Saving model based on improved SSIM!")
        dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
        torch.save(model.state_dict(), f'models/{dataset_name}_vqvae.pt') 
    else:
        print(f"Not saving model! Last best SSIM: {BEST_SSIM:.4f}")

    generate_samples()