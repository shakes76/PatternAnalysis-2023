"""For making predictions on the test dataset"""
from train import *
from dataset import *

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
    
def predict(model, dataloader, device):
    """Displays predictions of a handful of test images

    Args:
        model (<class 'modules.ViT'>): The trained model
        dataloader (torch.utils.data.DataLoader): The test data loader
        device (str): Device to predictions on
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store images and labels
    num_samples = 10
    all_images = []
    all_true_labels = []
    all_predicted_labels = []

    # Loop through the dataloader
    for images, true_labels in dataloader:
        # Move data to the specified device
        images, true_labels = images.to(device), true_labels.to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(images)
            predicted = outputs.argmax(1)

        # Store the images and labels
        all_images.extend(images.permute(0, 2, 3, 1).cpu().numpy())  # Permute to (batch, H, W, C)
        all_true_labels.extend(true_labels.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())

        if len(all_images) >= num_samples:
            break  # Exit the loop when 10 images have been collected

    # Plot the limited number of images, true labels, and model predictions in a single figure
    fig, axes = plt.subplots(int(num_samples / 5), int(num_samples / 2))
    for i, ax in enumerate(axes.ravel()):
        if i < len(all_images):
            image_data = all_images[i][:, :, 0]  # Extract the first channel
            true_label = all_true_labels[i]
            predicted_label = all_predicted_labels[i]

            ax.imshow(image_data, cmap='gray')
            ax.set_title(f"True: {true_label}\nPred: {predicted_label}", fontsize=7)
            ax.axis('off')
        else:
            ax.axis('off')

    # Set the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()