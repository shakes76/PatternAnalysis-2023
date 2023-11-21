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
N_EPOCHS = 30
BEST_EPOCH = 0
PRINT_INTERVAL = 100
DATASET_PATH = './OASIS'


NUM_WORKERS = 1
INPUT_DIM = 1
DIM = 256
K = 512
LAMDA = 1
LR = 1e-3
DEVICE = torch.device('cuda')
# Constants for determining the importance of SSIM and reconstruction loss
ALPHA = 0.5  # weight for SSIM, range [0, 1]
BETA = 1 - ALPHA  # weight for reconstruction loss
BEST_METRIC = -999  # initial value for the combination metric
BEST_SSIM = 0  # just for logging purposes
BEST_RECONS_LOSS = 999  # just for logging purposes
save_interval = 10
#train_losses = []
train_losses_epoch = [] # Will hold tuple of (reconstruction loss, VQ loss) per epoch
val_losses = [] # Will hold tuple of (reconstruction loss, VQ loss) per epoch for validation
ssim_scores = [] # Will hold SSIM scores for validation set


# Directories
Path('models5').mkdir(exist_ok=True)
Path('samples5').mkdir(exist_ok=True)

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
    total_loss_recons = 0.0
    total_loss_vq = 0.0
    num_batches = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time()
        x = x.to(DEVICE)
        opt.zero_grad()
        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad()
        loss_recons = F.mse_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True)
        total_loss_recons += loss_recons.item()
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
        #train_losses.append((log_px.item(), loss_recons.item(), loss_vq.item()))
        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.mean(train_losses_epoch[-PRINT_INTERVAL:], axis=0),
                time.time() - start_time
            ))
        total_loss_vq += loss_vq.item()
        num_batches += 1
    avg_loss_recons = total_loss_recons / num_batches
    avg_loss_vq = total_loss_vq / num_batches
    train_losses_epoch.append((avg_loss_recons, avg_loss_vq))
    print('Epoch Loss: Recons {:.4f}, VQ {:.4f}'.format(avg_loss_recons, avg_loss_vq))



def validate():
    global BEST_SSIM
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
    if avg_ssim > BEST_SSIM:
        BEST_SSIM = avg_ssim
    # Display the average validation loss
    mean_val_loss = np.asarray(val_loss).mean(0)
    print('\nValidation Loss: {}'.format(mean_val_loss))
    # print average ssim score
    print(f"Average Validation SSIM: {avg_ssim:.4f}")
    # Append the metrics to the lists
    val_losses.append(mean_val_loss)
    ssim_scores.append(avg_ssim)
    
    return np.asarray(val_loss).mean(0), avg_ssim  # return SSIM score and loss


def test():
    model.eval()  # Switch to evaluation mode
    ssim_accum = 0.0  # Accumulator for SSIM scores
    batch_count = 0   # Counter for batches
    with torch.no_grad():  # No gradient required for testing
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.to(DEVICE)
            x_tilde, _, _ = model(x)
            # Compute SSIM for the current batch and accumulate
            ssim_accum += compute_ssim(x, x_tilde)
            batch_count += 1
    # Calculate the average SSIM for all batches
    avg_ssim = ssim_accum / batch_count
    # print average ssim score for the test set
    print(f"Average Test SSIM: {avg_ssim:.4f}")
    return avg_ssim  # return SSIM score


def generate_samples(epoch):
    model.eval()  # make sure model is in eval mode
    x, _ = next(iter(test_loader)) # x, _ = test_loader.__iter__().next()
    x = x[:32].to(DEVICE)
    x_tilde, _, _ = model(x)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2
    dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
    save_image(
        images,
        f'samples5/vqvae_reconstructions_{epoch}.png',
        nrow=8
    )

    # Display the images
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'samples5/loss_plot_epoch{epoch}.png', bbox_inches='tight')
    plt.close()


def generate_sample_from_best_model(BEST_EPOCH):
    # Load the best model
    model.load_state_dict(torch.load(f'samples5/checkpoint_epoch{BEST_EPOCH}_vqvae.pt'))
    model.eval()

    # Get a sample from the test set  
    x, _ = next(iter(test_loader))
    x = x[:32].to(DEVICE)
    
    # Use the model to generate a sample
    x_tilde, _, _ = model(x)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2
    # Save the generated sample
    #save_image(x_tilde.cpu().data, 'samples2/best_model_sample.png')
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'samples5/best_model_sample.png', bbox_inches='tight')
    plt.close()


def plot_losses_and_scores():

    # Extract training losses for reconstruction and VQ
    train_recons_losses, train_vq_losses = zip(*train_losses_epoch)
    
    # Extract validation losses
    val_recons_losses, val_vq_losses = zip(*val_losses)
    
    epochs = range(len(train_recons_losses))
    # Plotting
    plt.figure(figsize=(12, 5))
    # Plot reconstruction losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_recons_losses, '-o', label='Training Recon Loss')
    plt.plot(epochs, val_recons_losses, '-o', label='Validation Recon Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # Plot VQ losses
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_vq_losses, '-o', label='Training VQ Loss')
    plt.plot(epochs, val_vq_losses, '-o', label='Validation VQ Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # Plot SSIM scores
    plt.subplot(1, 3, 3)
    plt.plot(epochs, ssim_scores, '-o', label='Validation SSIM')
    plt.xlabel("Epochs")
    plt.ylabel("SSIM Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig('samples5/loss_ssim_plot.png', bbox_inches='tight')
    plt.close()



for epoch in range(1, N_EPOCHS):
    print(f"Epoch {epoch}:")
    train()
    
    # Modify this line to unpack both loss and SSIM
    val_loss, val_ssim = validate()
    #test_ssim = test()
    # Calculate the combined metric
    combined_metric = ALPHA * val_ssim + BETA * (1 - val_loss[0])  # assuming lower reconstruction loss is better, thus the (1 - val_loss[0])
    # Check the combined metric for improvements
    if combined_metric > BEST_METRIC:                        
        BEST_METRIC = combined_metric
        BEST_SSIM = val_ssim
        BEST_EPOCH = epoch
        BEST_RECONS_LOSS = val_loss[0]
        print("Saving model based on improved combined metric!")
        dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
        # Save model and generate samples every 10 epochs

        torch.save(model.state_dict(), f'samples5/checkpoint_epoch{epoch}_vqvae.pt') 
    else:
        print(f"Not saving model! Last best combined metric: {BEST_METRIC:.4f}, SSIM: {BEST_SSIM:.4f}, Reconstruction Loss: {BEST_RECONS_LOSS:.4f}")

  
    # Generate samples at the end of each 5th epoch
    if epoch % save_interval == 0:
        generate_samples(epoch)
    
    
    # Step the scheduler to adjust learning rate
    scheduler.step()
# After all epochs are done
avg_test_ssim = test()  # Testing
print(f"Finished Training. Best SSIM on Validation set: {BEST_SSIM:.4f}. SSIM on Test set: {avg_test_ssim:.4f}")
plot_losses_and_scores()
# Generate a sample from the best model for inspection
generate_sample_from_best_model(BEST_EPOCH)