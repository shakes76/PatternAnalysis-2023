""" Show example usage of trained model. """

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def predict(model,
            dataloader,
            device,
            save_path):
    """ Display images from the dataloader at the given indices along with their true labels, predictions, and image names.

    Args:
        model (Module): The trained model
        dataloader (DataLoader): The dataloader containing the data
        device (str): Device to run the inference on
        n_samples (int, optional): Number of samples to display. Defaults to 10.
    """
    n_samples = 10
    labels = ["Alzheimer's", "Normal"]
    all_images = []
    all_labels_data = []
    
    # Loop over all data to gather images and labels
    for images, lbls in dataloader:
        all_images.append(images)
        all_labels_data.append(lbls)
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels_data, dim=0)

    # Randomly select n_samples indices
    indices = np.random.choice(len(all_images), n_samples, replace=False)

    # Extract specific images, labels, and image names based on indices
    selected_images = all_images[indices]
    selected_labels = all_labels[indices]
    
    # Move them to the specified device
    selected_images, selected_labels = selected_images.to(device), selected_labels.to(device)
    
    # Make predictions
    print(f"Making predictions...")
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        outputs = model(selected_images)
        probabilities = F.softmax(outputs, dim=1)     # Compute probabilities
        confidence, predicted = probabilities.max(1)  # Get the max probabilities and their indices
    
    # Plot the images with their true labels and predictions
    fig, axes = plt.subplots(int(n_samples/5), int(n_samples/2))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        # Display image
        image_data = selected_images[i][0].cpu().numpy()  # Directly extract the first channel
        ax.imshow(image_data, cmap='gray')
        ax.set_title(f"True: {labels[selected_labels[i].item()]}\n"
                    f"Pred: {labels[predicted[i].item()]}\n"
                    f"Conf: {confidence[i].item():.2f}",
                    fontsize=7)
        ax.axis('off')

    # Set the spacing between subplots
    plt.tight_layout()

    # Saving the plot
    plt.savefig(save_path + "/predictions.png")

    # Show the plot
    plt.show()