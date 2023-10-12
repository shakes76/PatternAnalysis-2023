# predict.py


import torch
from torchvision.utils import save_image
from modules import VectorQuantizedVAE, generate_sample_from_best_model

INPUT_DIM = 1
DIM = 256
K = 512
DEVICE = torch.device('cuda')

# Reading the best epoch from the file
with open("best_epoch.txt", "r") as file:
    BEST_EPOCH = int(file.readline().strip())

def predict(device, latent_dim, img_save_path):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device(device if use_cuda else "cpu")

    # Load the best model
    model_path = 'models2/checkpoint_epoch{}_vqvae.pt'.format(BEST_EPOCH) # using the BEST_EPOCH to load the best model
    model = VectorQuantizedVAE(INPUT_DIM, DIM, K).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate and save the image
    generated_img = generate_sample_from_best_model(model, device)
    save_image(generated_img, img_save_path)

if __name__ == "__main__":
    device = "cuda:0"
    latent_dim = 100
    img_save_path = './generated_image.png'
    predict(device, latent_dim, img_save_path)
"""
import torch
from modules import VectorQuantizedVAE, compute_ssim
from dataset import test_loader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from train import MODEL_PATH_TEMPLATE  # Importing the template string directly from train.py

# Load the best epoch value
with open("best_epoch.txt", "r") as file:
    BEST_EPOCH = int(file.read())

DEVICE = torch.device('cuda')
#MODEL_PATH = 'models/checkpoint_epoch{}_vqvae.pt'.format(BEST_EPOCH) # using the BEST_EPOCH to load the best model
MODEL_PATH = MODEL_PATH_TEMPLATE.format(BEST_EPOCH)

# Load the trained model
model = VectorQuantizedVAE(1, 256, 512).to(DEVICE)  # Assuming input_dim=1, dim=256, K=512 from your code
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def save_image(tensor, filename):
    img = tensor.cpu().clone().detach()
    img = img.numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.savefig(filename)


def predict(model, dataloader, device, output_dir="samples2"):
 #   
    for idx, (x, _) in enumerate(test_loader):
        x = x.to(DEVICE)
        x_tilde, _, _ = model(x)

        # Compute SSIM between original and reconstructed images
        ssim_score = compute_ssim(x, x_tilde)

        print(f"SSIM Score for batch {idx + 1}: {ssim_score:.4f}")
        
        # Save the reconstructed images for visualization
        for j, image in enumerate(x_tilde):
            save_image(image, f'samples3/reconstructed_{idx * len(x) + j + 1}.png')

        # For brevity, let's break after the first batch
        break
        #

    
    Generates predictions using the given model and dataloader. Saves the reconstructed images
    and their SSIM scores in the specified output directory.
    
    model.eval()  # Set the model to evaluation mode

    ssim_scores = []

    with torch.no_grad():  # No need to track gradients during prediction
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Calculate SSIM score
            inputs_np = inputs.cpu().numpy().squeeze(1)
            outputs_np = outputs.cpu().numpy().squeeze(1)
            score = ssim(inputs_np, outputs_np, data_range=outputs_np.max() - outputs_np.min())
            ssim_scores.append(score)

            # Save the reconstructed image
            save_path = os.path.join(output_dir, f"reconstructed_{i}.png")
            save_image(outputs, save_path)

    avg_ssim = np.mean(ssim_scores)
    print(f"Average SSIM Score: {avg_ssim}")

    # Save SSIM scores to a file for reference
    with open(os.path.join(output_dir, "ssim_scores.txt"), "w") as f:
        for i, score in enumerate(ssim_scores):
            f.write(f"Image {i}: {score}\n")
        f.write(f"\nAverage SSIM Score: {avg_ssim}")

    print(f"Saved reconstructed images and SSIM scores in {output_dir}")




if __name__ == '__main__':
    predict(model, dataloader, device)

"""
