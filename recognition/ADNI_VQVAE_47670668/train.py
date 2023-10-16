import torch
import torch.optim as optim

from dataset import train_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize values for incremental variance computation
mean = torch.zeros(1).float().to(device)
M2 = torch.zeros(1).float().to(device)
n = 0

# Loop through the DataLoader and compute incremental variance
for batch in train_dataloader:
    images = batch[0].float().to(device) / 255.0
    batch_mean = images.mean()
    n_batch = images.numel()
    n += n_batch

    delta = batch_mean - mean
    mean += delta * n_batch / n
    M2 += delta * (batch_mean - mean) * n_batch


# Compute variance
if n < 2:
    train_data_variance = float('nan')
else:
    train_data_variance = M2 / (n - 1)

