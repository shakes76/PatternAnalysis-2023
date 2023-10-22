import torch  # Import PyTorch for deep learning functionalities.
import matplotlib.pyplot as plt  # for plotting and visualization.
import numpy as np  # for numerical computations.
from torch.utils.data import DataLoader  # for creating manageable batches from datasets.

# Get functions from other file.
from dataset import ISICDataset, get_transform, get_mask_transform
from modules import ImprovedUNet

# Determine if CUDA (GPU support) is available, use it, otherwise default to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Predict: {torch.cuda.is_available()}") # Display whether CUDA is available.


def imshow(inp, title=None, gray=False, ax=None):
    """A helper function to display images with certain adjustments."""

    # If the input is a Torch tensor, we need to convert it to a NumPy array first
    if isinstance(inp, torch.Tensor):
        inp = inp.numpy()

    # If image has an extra dimension of size 1, remove that dimension to simplify the array.
    if inp.shape[0] == 1:
        inp = np.squeeze(inp, axis=0)

    # Set the colormap to 'gray' if the gray flag is True, otherwise it will be default (None).
    if gray:
        cmap = 'gray'
    else:
        cmap = None

    # Image array has three dimensions, assume it's in channels, height, width.
    # Then rearrange this to height, width, channels for proper display with imshow.
    if len(inp.shape) == 3:
        inp = inp.transpose((1, 2, 0))  # Reordering dimensions.

        # Reverting to bring back the original colors for display.
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        inp = std * inp + mean  # De-normalizing.

        # Clipping the values to be between 0 and 1, as imshow expects values in this range for a float array.
        inp = np.clip(inp, 0, 1)

    if ax is None:
        plt.imshow(inp, cmap=cmap)  # Creating a new plot.
        if title is not None:
            plt.title(title)  # Setting the title of the plot.
    else:
        ax.imshow(inp, cmap=cmap)  # Using the provided axis for the plot.
        ax.set_title(title)  # Setting the title of the subplot.

    # A small pause is added to allow time for the plot to be created.
    plt.pause(0.001)


def my_collate(batch):
    """Custom collate function to handle samples with missing masks."""

    # This function ensures that the data loader doesn't crash if it encounters a sample without a mask.
    new_batch = list(filter(lambda x: x[1] is not None, batch))

    # Check if there are any valid samples in the batch, if all masks were None, we skip this batch.
    if len(new_batch) == 0:  #
        print("Warning: Empty batch encountered. All masks were None.")
        return None

    # If there's at least one valid sample, we create a batch for processing.
    return torch.utils.data.dataloader.default_collate(new_batch)



def predict():
    """Function to load the data, run it through the model, and display the model's predictions."""

    # Define transformations for the images and masks.
    img_transform = get_transform()
    mask_transform = get_mask_transform()

    test_dataset = ISICDataset(
        image_dir="ISIC2018_Task1-2_Training_Input_x2",
        mask_dir="ISIC2018_Task1_Training_GroundTruth_x2",
        img_transform=img_transform,
        mask_transform=mask_transform,
        img_size=(1024, 1024)  # dimensions to which images will be resized.
    )

    # Setting up the data loader for batch processing.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate)

    # Initializing the model, loading its weights, and setting it to evaluation mode.
    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load("plot_checkpoint.pth"))
    model.eval()  # Switch to evaluation mode to prevent changes to model parameters.

    num_images_to_display = 5  # Number of images we want to run through the model and display.

    # Creating an iterator from the data loader.
    data_iter = iter(test_loader)

    # Loop through batches for predictions.
    for _ in range(num_images_to_display):
        # This loop retrieves batches of images from the data loader, applies the model, and visualizes the results.
        # It skips any batches that were found to be invalid (no masks).
        try:
            batch = next(data_iter)
            if batch is None:
                continue  # Skip this loop iteration if batch is empty.

            images, true_masks = batch  # Unpack batch.

        except StopIteration:
            break  # If gone through the whole dataset, exit the loop.

        # Move images and masks to the device that's being used from GPU or CPU.
        images = images.to(device)
        true_masks = true_masks.to(device)

        # Get the model's predictions for the masks.
        with torch.no_grad():
            outputs = model(images)

        # Extracting the predicted masks and moving them back to the CPU for visualization.
        predicted_masks = outputs.data.cpu().numpy()
        true_masks = true_masks.cpu().numpy()
        images = images.cpu()

        # Loop through and display each image and its corresponding masks
        for image, pred_mask, true_mask in zip(images, predicted_masks, true_masks):
            plt.subplot(1, 3, 1)
            imshow(image, title='Original Image')  # show the original image.

            plt.subplot(1, 3, 2)
            imshow(np.squeeze(pred_mask), title='Predicted Mask', gray=True)  # show the predicted mask.

            plt.subplot(1, 3, 3)
            imshow(np.squeeze(true_mask), title='True Mask', gray=True)  # show the true mask.

        plt.show()  # Display the plotted images.


if __name__ == "__main__":
    predict()  # Run the prediction function when the script is executed directly.
    plt.show(block=True)  # Keep the plot open.
