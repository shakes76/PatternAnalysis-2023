from utils import *
from dataset import *
import torch
import matplotlib.pyplot as plt

# -------
# Initialise device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

print("> Loading model")
model = torch.load(model_path)

train_loader = generate_train_loader()

column_labels = ["Original Image", "Downsampled Image", "Model Prediction"]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 7))

for ax, col in zip(axes[0], column_labels):
    ax.set_title(col)

print("> Generating predictions for first 3 images")

for i, (images, _) in enumerate(train_loader):

    original_1 = images[0][0].cpu().detach().numpy()
    original_2 = images[1][0].cpu().detach().numpy()
    original_3 = images[2][0].cpu().detach().numpy()

    low_res_images = resize_tensor(images)
    low_res_images = low_res_images.to(device)

    downsample_1 = low_res_images[0][0].cpu().detach().numpy()
    downsample_2 = low_res_images[1][0].cpu().detach().numpy()
    downsample_3 = low_res_images[2][0].cpu().detach().numpy()

    # Forward pass
    outputs = model(low_res_images)
    outputs = outputs.to(device)

    output_1 = outputs[0][0].cpu().detach().numpy() 
    output_2 = outputs[1][0].cpu().detach().numpy()
    output_3 = outputs[2][0].cpu().detach().numpy()

    axes[0, 0].imshow(original_1, cmap='gray')
    axes[1, 0].imshow(original_2, cmap='gray')
    axes[2, 0].imshow(original_3, cmap='gray')

    axes[0, 1].imshow(downsample_1, cmap='gray')
    axes[1, 1].imshow(downsample_2, cmap='gray')
    axes[2, 1].imshow(downsample_3, cmap='gray')

    axes[0, 2].imshow(output_1, cmap='gray')
    axes[1, 2].imshow(output_2, cmap='gray')
    axes[2, 2].imshow(output_3, cmap='gray')


    plt.show()
    break



