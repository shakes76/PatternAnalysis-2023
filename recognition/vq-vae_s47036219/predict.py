import torch
from modules import VQVAE, ssim
from dataset import get_dataloaders
from train import SSIM_WEIGHT, L2_WEIGHT, BATCH_SIZE, train_new_model, path_to_training_folder, path_to_test_folder
import matplotlib
import matplotlib.pyplot as plt
import os

def evaluate(test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE().to(device)
    model.load_state_dict(torch.load('vqvae.pth'))
    model.eval()
    print("loaded")
    highest_ssim_val = float('-inf')  # Initialize with negative infinity
    lowest_ssim_val = float('inf')  # Initialize with positive infinity
    highest_ssim_img = None
    highest_ssim_recon = None
    lowest_ssim_img = None
    lowest_ssim_recon = None
    
    val_losses = []
    ssim_sum = 0  # To keep track of sum of all SSIM values
    total_images = 0  # To keep track of total number of images processed

    with torch.no_grad():
        for i, (img, _) in enumerate(test_loader):
            img = img.to(device)
            
            # Validation forward pass
            z = model.encoder(img)
            z = model.conv1(z)
            z_q = model.vector_quantizer(z)
            recon = model.decoder(z_q)

            # Validation losses
            l2_loss = ((recon - img) ** 2).sum()
            ssim_loss = 1 - ssim(img, recon)
            loss = L2_WEIGHT * l2_loss + SSIM_WEIGHT * ssim_loss
            val_losses.append(loss.item())

            # Calculate SSIM
            ssim_val = ssim(img, recon).item()
            ssim_sum += ssim_val  # Add SSIM value to the sum
            total_images += img.size(0)  # Increase the total number of images processed

            #print(f'SSIM: {ssim_val}')  # Output SSIM value

            # Update highest and lowest SSIM values and corresponding images
            if ssim_val > highest_ssim_val:
                highest_ssim_val = ssim_val
                highest_ssim_img = img.cpu().numpy().squeeze(1)
                highest_ssim_recon = recon.cpu().numpy().squeeze(1)
            
            if ssim_val < lowest_ssim_val:
                lowest_ssim_val = ssim_val
                lowest_ssim_img = img.cpu().numpy().squeeze(1)
                lowest_ssim_recon = recon.cpu().numpy().squeeze(1)

    mean_ssim = ssim_sum / total_images
    print(f'Mean SSIM: {mean_ssim}')  # Output mean SSIM value

    # Output images with the highest and lowest SSIM values
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.title(f'Original Highest SSIM: {highest_ssim_val}')
    plt.imshow(highest_ssim_img[0], cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title('Reconstructed')
    plt.imshow(highest_ssim_recon[0], cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title(f'Original Lowest SSIM: {lowest_ssim_val}')
    plt.imshow(lowest_ssim_img[0], cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title('Reconstructed')
    plt.imshow(lowest_ssim_recon[0], cmap='gray')

    plt.tight_layout()
    plt.show()
    
def main():
    weight_file_path = "vqvae.pth"
    
    train, validate, test = get_dataloaders(path_to_training_folder, path_to_training_folder, BATCH_SIZE)

    if os.path.exists(weight_file_path):
        print("Weights exist -> Evaluating Model...")
        evaluate(test)
        
    else:
        print(f"Weight file {weight_file_path} does not exist.")
        print("Training model now...")
        train_new_model(train, validate)


    
if __name__ == "__main__":
    main()