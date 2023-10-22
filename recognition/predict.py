import torch # Import PyTorch for deep learning functionalities.
import torch.nn.functional as F # Import functional interface
import torch.optim as optim # Import optimizer support from PyTorch (Adam).
import matplotlib.pyplot as plt # For plotting and visualization.

from torch.utils.data import DataLoader, random_split # For batch loading of datasets, and random_split for splitting datasets into training and validation.
from torchvision import transforms # Import transforms for image preprocessing.
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import learning rate scheduler which reduces LR based on some criterion.

# Get functions from other file.
from dataset import ISICDataset, get_transform, get_mask_transform
from modules import ImprovedUNet, device
from predict import predict

# Determine if CUDA (GPU support) is available, use it, otherwise default to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Train: {torch.cuda.is_available()}") # Display whether CUDA is available.


def dice_loss(pred, target, smooth=1.):
    """Compute the Dice loss, which is particularly useful for imbalanced segmentation problems."""

    # Apply the sigmoid activation function to the predictions.
    pred = torch.sigmoid(pred)

    # Flatten the predictions and the target labels in order to compute the Dice coefficient.
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute the intersection of the prediction and target tensors.
    intersection = (pred * target).sum()

    # Compute the Dice coefficient.
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # Return the Dice loss.
    return 1 - dice


def combined_loss(pred, target, alpha=0.5):
    """Calculate the combined loss, which is a weighted sum of binary cross entropy loss and Dice loss."""

    # Calculate the Binary Cross Entropy loss.
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Calculate the Dice loss.
    dice = dice_loss(pred, target)

    # Return the weighted sum of the computed BCE and Dice losses.
    return alpha * bce + (1 - alpha) * dice


def evaluate_dsc(loader, model):
    """ Evaluate the model on a dataset using the Dice Similarity Coefficient (DSC)."""

    # Set the model in evaluation mode.
    model.eval()
    all_dscs = []

    # Disable gradient calculations during forward passes.
    with torch.no_grad():
        # Iterate over the dataset.
        for images, masks in loader:
            # Transfer images and masks to the device that is used from GPU or CPU.
            images, masks = images.to(device), masks.to(device)

            # Make predictions.
            outputs = model(images)

            # Compute the Dice loss.
            loss = dice_loss(outputs, masks)

            # Calculate the Dice Similarity Coefficient.
            dice_coefficient = 1 - loss.item()

            # Append the coefficient to the list for all samples.
            all_dscs.append(dice_coefficient)

    # Return the list of Dice Similarity Coefficients for further analysis.
    return all_dscs


def get_transform():
    """Constructs the transform needed for the input images."""

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_mask_transform():
    """Constructs the transform needed for the mask images."""

    return transforms.Compose([
        transforms.ToTensor()
    ])


def train():
    """The main function to handle the training of the model."""

    # Step 1: Prepare the transformations for image and mask.
    img_transform = get_transform()
    mask_transform = get_mask_transform()

    # Step 2: Load the ISIC2018 dataset.
    full_dataset = ISICDataset(
        image_dir="ISIC2018_Task1-2_Training_Input_x2",  # Path containing training images.
        mask_dir="ISIC2018_Task1_Training_GroundTruth_x2",  # Path containing ground truth masks.
        img_transform=img_transform,  # Transformations applied to the training images.
        mask_transform=mask_transform,  # Transformations applied to the masks.
        img_size=(384, 512)  # Resized the target image.
    )

    # Step 3: Split the dataset into training and test subsets.
    train_size = int(0.85 * len(full_dataset))  # 85% of the dataset is used for training.
    test_size = len(full_dataset) - train_size  # The rest is used for testing/validation.
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])  # Performing the split.

    # Step 4: Create data loaders. These are sophisticated data handling tools.
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # DataLoader for the training dataset.
    val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # DataLoader for the test/validation dataset.

    # Step 5: Define the neural network model, optimizer, and learning rate scheduler.
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)  # Initialize the model and move it to the CPU or GPU.
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Adam optimizer used with learning rate and weight decay set.
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # Scheduler for adjusting the learning rate.

    # Step 6: Define the number of epochs for which the model will be trained .
    num_epochs = 25  # Set number of training.
    train_losses, val_losses = [], []  # Lists to track the loss metrics for training and validation phases.
    avg_train_dscs, avg_val_dscs = [], []  # Lists to track the average DSC for training and validation phases.

    # Step 7: Initialize the strategy for gradient accumulation.
    accumulation_steps = 4  # The number of steps/batches over which the gradient is accumulated.
    optimizer.zero_grad()  # Ensuring the gradients are zero.

    for epoch in range(num_epochs):
        # Print current epoch number out of total epochs which is easier for me to keep track of epoch process.
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Set the model to "train" mode.
        model.train()

        # Initialize variables to accumulate the losses and dice similarity coefficients within this epoch.
        running_loss, running_dsc = 0.0, 0.0

        # Training phase.
        for i, (inputs, labels) in enumerate(train_loader):
            # Move the inputs and labels to CPU or GPU that will be used for computations.
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear the gradients from the previous iteration.
            optimizer.zero_grad()

            # Forward pass: compute the predictions by passing the inputs through the model.
            outputs = model(inputs)

            # Compute the loss based on the outputs and actual labels.
            loss = combined_loss(outputs, labels)

            # Backward pass: compute the gradients of the loss.
            loss.backward()

            # Update the model’s parameters based on the gradients computed during the backward pass.
            optimizer.step()

            # Compute the Dice Similarity Coefficient (DSC) for evaluation.
            dsc = 1 - dice_loss(outputs, labels).item()

            # Accumulate the DSC and the loss over the epoch.
            running_dsc += dsc
            running_loss += loss.item()

            # Delay the optimizer step, so it help with my device memory.
            if (i + 1) % accumulation_steps == 0:
                # Perform the actual model parameters update.
                optimizer.step()
                # Remove the gradients.
                optimizer.zero_grad()

        # Calculate the average losses and DSC for this epoch.
        avg_train_loss = running_loss / len(train_loader)
        avg_train_dsc = running_dsc / len(train_loader)

        # Append the average loss and DSC for this epoch to the lists for later visualization and plot.
        train_losses.append(avg_train_loss)
        avg_train_dscs.append(avg_train_dsc)

        model.eval()  # Set the model to evaluation mode.
        val_loss, val_dsc = 0.0, 0.0  # Initialize counters for the validation loss and DSC.

        with torch.no_grad():  # Remove gradients for validation so it save memory.
            for images, masks in val_loader:
                # Move the images and masks to CPU or GPU.
                images, masks = images.to(device), masks.to(device)

                # Forward pass: Compute the predictions by passing images through model.
                outputs = model(images)

                # Compute the loss between the predicted outputs and the actual masks.
                loss = combined_loss(outputs, masks)

                # Accumulate the validation loss over all the batches for this epoch.
                val_loss += loss.item()

                # Calculate and accumulate the Dice Similarity Coefficient for the validation set.
                dice_val = 1 - dice_loss(outputs, masks).item()
                val_dsc += dice_val

        # Calculate the average validation loss and DSC for this epoch.
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dsc = val_dsc / len(val_loader)

        # Append the averages to maintain a history for further analysis and plotting after all epochs.
        val_losses.append(avg_val_loss)
        avg_val_dscs.append(avg_val_dsc)

        # Display the average training and validation loss/DSC after each epoch.
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train DSC: {avg_train_dsc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val DSC: {avg_val_dsc:.4f}")

        # Adjust the learning rate based on the validation loss.
        scheduler.step(avg_val_loss)

    # Prepares to plot the training and validation loss and DSC.
    plt.figure(figsize=(15, 10))

    # Plotting the training loss.
    plt.subplot(2, 2, 1)

    # 'range(1, num_epochs + 1)' creates X-axis representing each epoch.
    # 'train_losses' contains the loss value for each epoch, forming the Y-axis.
    
    # Plotting the Training Loss.
    plt.plot(range(1, num_epochs + 1), train_losses, '-o', label='Training Loss')
    plt.title('Training Loss')  # Title of the plot.
    plt.xlabel('Epochs')  # Label for the X-axis.
    plt.ylabel('Loss')  # Label for the Y-axis.
    plt.legend()  # Add a legend to help identify each line on the plot.

    # Plotting the training Dice Similarity Coefficient (DSC).
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), avg_train_dscs, '-o', label='Training DSC')
    plt.title('Training Dice Similarity Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()

    # Plotting the validation loss.
    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), val_losses, '-o', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting the validation DSC.
    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), avg_val_dscs, '-o', label='Validation DSC')
    plt.title('Validation Dice Similarity Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()

    # Adjust the subplots to fit in the figure.
    plt.tight_layout()
    # Renders the complete figure.
    plt.draw()
    # A short pause to ensure the plot gets rendered.
    plt.pause(0.001)

    # Saving the model's learned parameters.
    torch.save(model.state_dict(), "plot_checkpoint.pth")
    plt.ioff()
    # Display the figure as a blocking call - the script will stop here until the plot is closed.
    plt.show(block=True)

if __name__ == "__main__":
    # Call the 'train' function from train.py to start the training process.
    train()
    print("Training completed. Starting prediction phase...")
    # Call the 'predict' function from predict.py.
    predict()
