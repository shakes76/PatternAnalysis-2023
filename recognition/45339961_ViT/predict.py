""" Show example usage of trained model. """

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def test(model, device, test_loader, criterion):
    """ Test the model on the test set.

    Args:
        model (Module): The trained model
        device (str): Device to run the inference on
        test_loader (Dataloader): The dataloader containing the test data
        criterion (Criterion): Loss function
    """

    # Set the model to evaluation mode
    model.eval()  
    
    # Test loop
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss = criterion(y_hat, y)

            test_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        test_loss /= len(test_loader)  # Calculate average test loss
        accuracy = 100 * correct / total  # Calculate accuracy
        
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {accuracy:.2f}%")

def predict(model, dataloader, device, n_samples=10):
    """ Display images from the dataloader at the given indices along with their true labels, predictions, and image names.

    Args:
        model (Module): The trained model
        dataloader (DataLoader): The dataloader containing the data
        device (str): Device to run the inference on
        n_samples (int, optional): Number of samples to display. Defaults to 10.
    """
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
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        outputs = model(selected_images)
        probabilities = F.softmax(outputs, dim=1)  # Compute probabilities
        confidence, predicted = probabilities.max(1)  # Get the max probabilities and their indices
    
    # Plot the images with their true labels and predictions
    fig, axes = plt.subplots(2, 5)
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

    plt.tight_layout()
    plt.show()