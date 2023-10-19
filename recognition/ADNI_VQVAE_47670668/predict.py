import torch
from pytorch_msssim import ssim

from dataset import test_loader
from train import visualize_reconstructions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('model_path.pth').to(device)


# Produce reconstructions for test set
model.eval()  # Set the model to evaluation mode
average_ssim = 0

with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device)

        # Forward pass
        output_dict = model(inputs)

        # Extract the reconstructed images tensor
        reconstructed_images = output_dict['x_recon']

        # Calculate SSIM
        current_ssim = ssim(reconstructed_images, inputs, data_range=1.0)
        average_ssim += current_ssim.item()

        # Visualize the first batch's reconstructions for demonstration
        if batch_idx == 0:
            visualize_reconstructions(inputs, reconstructed_images)

# Average the SSIM over all batches
average_ssim = average_ssim / len(test_loader)

print(f"Average SSIM on test set: {average_ssim:.4f}")  