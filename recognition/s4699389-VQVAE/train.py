import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from dataset import OASISDataLoader
import modules

VQVAE_PATH = "./vqvae_model.txt"
TRAIN_OUTPUT_PATH = "./train_output.out"
save_model = True # Change depending if you want to save model to file

# Torch configuration
seed = 42
torch.manual_seed(seed)
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print("Name: ", torch.cuda.get_device_name(0))

# Hyperparameters
num_epochs = 2  # Change to desired epochs

batch_size = 32
learning_rate = 0.0002
commitment_cost = 0.25
num_hiddens = 128
num_residual_hiddens = 32
num_channels = 1
embedding_dim = 64
num_embeddings = 512

train_loader, _, val_loader = OASISDataLoader(batch_size=batch_size).get_dataloaders()

# Calculate variance
mean = 0.0
mean_sq = 0.0
count = 0

for index, data in enumerate(train_loader):
    mean = data.sum()
    mean_sq = mean_sq + (data ** 2).sum()
    count += np.prod(data.shape)

total_mean = mean/count
total_var = (mean_sq / count) - (total_mean ** 2)
data_variance = float(total_var.item()) # 0.68


model = modules.VQVAE(num_channels, num_hiddens, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# Training loop
epoch_train_loss = []
epoch_validation_loss = []
epoch_ssim = []

train_error = []
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    model.train()
    train_loss = 0

    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon = model(data)

        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_error.append(recon_error.item())

    train_loss = np.mean(train_error[-300:])
    epoch_train_loss.append(train_loss)
    print('training_loss: %.3f' % train_loss)

    # Evaluate on the validation dataset
    model.eval()
    val_loss = 0
    validation_ssim = []
    with torch.no_grad():
        for j, val_data in enumerate(val_loader):
            val_data = val_data.to(device)
            vq_loss, data_recon = model(val_data)

            real_img = val_data.view(-1, 1, 128, 128).detach()
            decoded_img = data_recon.view(-1, 1, 128, 128).to(device).detach()
            ssim_val = ssim(decoded_img, real_img, data_range=1.0).item()
            validation_ssim.append(ssim_val)
            recon_error = F.mse_loss(data_recon, val_data) / data_variance
            val_loss += recon_error + vq_loss

    average_val_loss = val_loss / len(val_loader)
    average_ssim = np.mean(ssim_val)
    print('validation_loss: %.3f' % average_val_loss)
    epoch_validation_loss.append(average_val_loss.item())
    print('average_ssim: %.3f' % average_ssim)
    epoch_ssim.append(average_ssim)
    print()

# Save model
if save_model:
    torch.save(model, VQVAE_PATH)

# Save training data
with open(TRAIN_OUTPUT_PATH, 'w') as file:
    for i in range(num_epochs):
        file.write(f"Epoch: {i}\n")
        file.write(f"training_loss: {str(epoch_train_loss[i])}\n")
        file.write(f"validation_loss: {str(epoch_validation_loss[i])}\n")
        file.write(f"average_ssim: {str(epoch_ssim[i])}\n")
        file.write("\n")