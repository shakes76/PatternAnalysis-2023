# train.py

import torch
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import cycle
import torch.nn.functional as F
from torch.distributions.normal import Normal
from modules import VectorQuantizedVAE, compute_ssim, weights_init, to_scalar, plot_losses_and_scores, generate_samples
from dataset import train_loader, val_loader, test_loader

# Constants
BATCH_SIZE = 32
#N_EPOCHS = 401
N_EPOCHS = 346
PRINT_INTERVAL = 100
DATASET_PATH = './OASIS'
NUM_WORKERS = 1
INPUT_DIM = 1
DIM = 256
K = 512
LAMDA = 1
LR = 1e-3
DEVICE = torch.device('cuda')

# Global best epoch and model path
BEST_EPOCH = 0
MODEL_PATH_TEMPLATE = 'samples6/checkpoint_epoch{}_vqvae.pt'

# Constants for determining the importance of SSIM and reconstruction loss
ALPHA = 0.5  # weight for SSIM, range [0, 1]
BETA = 1 - ALPHA  # weight for reconstruction loss
BEST_METRIC = -999  # initial value for the combination metric
BEST_SSIM = 0  # just for logging purposes
BEST_RECONS_LOSS = 999  # just for logging purposes
save_interval = 15


#directory creation
train_losses_epoch = []
val_losses = []
ssim_scores = []

# Directories
#Path('models2').mkdir(exist_ok=True)
Path('samples6').mkdir(exist_ok=True)
Path('models6').mkdir(exist_ok=True)

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
    if num_batches > 0:
        avg_loss_recons = total_loss_recons / num_batches
        avg_loss_vq = total_loss_vq / num_batches
    else:
        avg_loss_recons = 0
        avg_loss_vq = 0
    

    train_losses_epoch.append((avg_loss_recons, avg_loss_vq))
    print('Epoch Loss: Recons {:.4f}, VQ {:.4f}'.format(avg_loss_recons, avg_loss_vq))
        

def validate():
    model.eval()  # Switch to evaluation mode
    val_loss = []
    ssim_accum = 0.0  # Accumulator for SSIM scores
    batch_count = 0   # Counter for batches
    total_loss_recons = 0.0
    total_loss_vq = 0.0
    num_batches = 0
    with torch.no_grad():  # No gradient required for validation
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(DEVICE)
            x_tilde, z_e_x, z_q_x = model(x)
            loss_recons = F.mse_loss(x_tilde, x)
            total_loss_recons += loss_recons.item()
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            total_loss_vq += loss_vq.item()
            val_loss.append(to_scalar([loss_recons, loss_vq]))

            # Compute SSIM for the current batch and accumulate
            ssim_current = compute_ssim(x, x_tilde)
            ssim_accum += ssim_current
            
            batch_count += 1
            num_batches += 1
    if num_batches > 0:
        avg_loss_recons = total_loss_recons / num_batches
        avg_loss_vq = total_loss_vq / num_batches
        avg_ssim = ssim_accum / num_batches
    else:
        avg_loss_recons = 0
        avg_loss_vq = 0
        avg_ssim = 0
        
    val_losses.append((avg_loss_recons, avg_loss_vq))
    ssim_scores.append(avg_ssim)
    combined_metric = ALPHA * avg_ssim - BETA * avg_loss_recons
        
    return combined_metric
    

    
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

train_losses_epoch = []
total_loss_recons = 0.0


if __name__ == '__main__':
    
    for epoch in range(1, N_EPOCHS):
        print(f"Epoch {epoch}:")
        train()
       
        # Calculate the combined metric
        combined_metric = validate()

        # Check the combined metric for improvements
        if combined_metric > BEST_METRIC:                        
            BEST_METRIC = combined_metric
            BEST_EPOCH = epoch
            print("Saving model based on improved combined metric!")
            dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
            torch.save(model.state_dict(), f'samples6/checkpoint_epoch{BEST_EPOCH}_vqvae.pt')
            
            with open("best_epoch.txt", "w") as file:
                file.write(str(BEST_EPOCH))

            # Log the best reconstruction loss and SSIM
            BEST_SSIM = ssim_scores[-1]
            BEST_RECONS_LOSS = val_losses[-1][0]
        
        else:
            print(f"Not saving model! Last best combined metric: {BEST_METRIC:.4f}, SSIM: {BEST_SSIM:.4f}, Reconstruction Loss: {BEST_RECONS_LOSS:.4f}")
        if epoch == 1:
            generate_samples(model, test_loader, epoch) # prints first constructed image for comparison
        # Generate samples at the end of each 5th epoch
        if epoch % save_interval == 0:
            generate_samples(model, test_loader, epoch)
        
        # Step the scheduler to adjust learning rate
        scheduler.step()
    # Load the best model before testing
    model.load_state_dict(torch.load(MODEL_PATH_TEMPLATE.format(BEST_EPOCH)))
    # Testing the model after training
    test_ssim = test()
    print(f"Average SSIM on Test Set: {test_ssim:.4f}")

    plot_losses_and_scores(train_losses_epoch, val_losses, ssim_scores)

    # Saving the best epoch for future reference (to be used in predict.py)
    with open("best_epoch.txt", "w") as file:
        file.write(str(BEST_EPOCH))
    
