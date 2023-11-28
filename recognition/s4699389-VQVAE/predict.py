import matplotlib.pyplot as plt
import torch
import torchvision
from dataset import OASISDataLoader
import parameters
import os

# Check if CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained VQ-VAE model
model = torch.load(parameters.VQVAE_PATH)
_, test_loader, _ = OASISDataLoader(batch_size=parameters.batch_size) \
    .get_dataloaders()

# Image representations
for i, test_data in enumerate(test_loader):
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    for x in ax.ravel():
        x.set_axis_off()

    # Prepare images for visualization
    test_real = test_data[0]
    test_real = test_real.to(device).view(-1, 1, 128, 128).detach()
    real_grid = torchvision.utils.make_grid(test_real, normalize=True)
    real_grid = real_grid.to("cpu").permute(1, 2, 0)

    _, decoded_img = model(test_real)
    decoded_img = decoded_img.view(-1, 1, 128, 128).to(device).detach()
    decoded_grid = torchvision.utils.make_grid(decoded_img, normalize=True)
    decoded_grid = decoded_grid.to("cpu").permute(1, 2, 0)

    # Get the codebook representation
    pre_conv = (model.pre_vq_conv(model.encoder(test_real)))
    _, test_quantized, _, indices = model.vq(pre_conv)
    encoding = indices.view(32, 32)
    encoding = encoding.to('cpu')
    encoding = encoding.detach().numpy()

    # Display images and codebook representation
    ax[0].imshow(real_grid)
    ax[0].title.set_text("Real Image")
    ax[1].imshow(encoding)
    ax[1].title.set_text("Codebook Representation")
    ax[2].imshow(decoded_grid)
    ax[2].title.set_text("Decoded Image")

    # Save images if specified
    if parameters.save_figure:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'image_reconstructions/')
        file_name = f"image_representation_{i}.png"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(results_dir + file_name)
    plt.show()

    # Exit the loop after displaying one figure if specified
    if parameters.show_one_figure:
        break


# Plot graphs
epochs = []
training_losses = []
validation_losses = []
average_ssims = []

# Read the data from the text file
with open(parameters.TRAIN_OUTPUT_PATH, "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 5):
        epochs.append(int(lines[i].split()[1]))
        training_losses.append(float(lines[i+1].split()[1]))
        validation_losses.append(float(lines[i+2].split()[1]))
        average_ssims.append(float(lines[i+3].split()[1]))

# Create the plot for training and validation losses
if parameters.train_loss:
    plt.figure(figsize=(8, 5))
    plt.ylim(0, training_losses[0])
    plt.plot(epochs, training_losses, linestyle='-', label="Training loss")
    plt.plot(epochs, validation_losses, linestyle='-', label="Validation loss")
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)

    # Save the loss plot if specified
    if parameters.save_graphs:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'model_plots/')
        file_name = "loss_plot.png"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(results_dir + file_name)
    plt.show()

# Create the plot for average SSIM values
if parameters.mean_ssim:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, average_ssims, linestyle='-', color='red',
             label="Average SSIM per epoch")
    plt.legend()
    plt.title('Average SSIM vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average SSIM per epoch')
    plt.grid(False)

    # Save the SSIM plot if specified
    if parameters.save_graphs:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'model_plots/')
        file_name = "average_ssim.png"
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(results_dir + file_name)
    plt.show()
