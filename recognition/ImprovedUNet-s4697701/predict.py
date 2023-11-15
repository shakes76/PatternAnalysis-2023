import torch
import torch.nn.functional as F  # For calculating the loss
import matplotlib.pyplot as plt
from dataset import ISICDataset, get_transform
from modules import UNet2D

def dice_coefficient(predicted, target):
    """
    Compute the Dice coefficient, a measure for the overlap between two samples.
    
    Parameters:
    - predicted (Tensor): The predicted segmentation mask.
    - target (Tensor): The ground truth segmentation mask.
    
    Returns:
    - float: Dice coefficient.
    """
    smooth = 1.0  # To avoid division by zero
    predicted_flat = predicted.view(-1)
    target_flat = target.view(-1)
    intersection = (predicted_flat * target_flat).sum()
    return (2. * intersection + smooth) / (predicted_flat.sum() + target_flat.sum() + smooth)

def predict_and_evaluate(model, sample, device):
    """
    Generate predictions using the model and evaluate them against the ground truth.
    
    Parameters:
    - model (UNet2D): The model used for prediction.
    - sample (dict): A dictionary containing the input image and target mask.
    - device (torch.device): Device to which Tensors should be moved before computation.
    
    Returns:
    - Tensor: Predicted mask.
    - float: Binary cross entropy loss.
    - float: Dice coefficient.
    """
    model.eval()
    image = sample['image'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    pred_mask = (output > 0.5).float()
    loss = F.binary_cross_entropy_with_logits(output, mask)
    dice = dice_coefficient(pred_mask, mask)
    return pred_mask.squeeze().cpu(), loss.item(), dice.item()

def plot_results(samples, pred_masks, num_samples):
    """
    Visualize and save the input images, ground truth masks, and predicted masks.
    
    Parameters:
    - samples (list): List of samples containing images and masks.
    - pred_masks (list): List of predicted masks.
    - num_samples (int): Number of samples to display.
    """
    fig, ax = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples)) 
    for i in range(num_samples):
        ax[i, 0].imshow(samples[i]['image'].permute(1, 2, 0))
        ax[i, 1].imshow(samples[i]['mask'].squeeze(), cmap="gray")
        ax[i, 2].imshow(pred_masks[i].squeeze(), cmap="gray")
        if i == 0: 
            ax[i, 0].set_title("Input Image")
            ax[i, 1].set_title("Actual Mask")
            ax[i, 2].set_title("Predicted Mask")
    plt.tight_layout()
    plt.savefig("test_results_plot.png", dpi=300)  # Save the figure
    plt.show()

def main():
    """
    Main function to run the entire pipeline of loading the model, 
    generating predictions, evaluating them, and visualizing the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model_epoch_25.pth"
    model = UNet2D(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_dataset = ISICDataset(dataset_type='test', transform=get_transform())
    
    num_samples = 5
    total_loss = 0.0
    total_dice = 0.0
    
    samples = []
    pred_masks = []
    for i in range(num_samples):
        sample = test_dataset[i]
        pred_mask, loss, dice = predict_and_evaluate(model, sample, device)
        
        samples.append(sample)
        pred_masks.append(pred_mask)
        
        total_loss += loss
        total_dice += dice
        
    # Plot the results
    plot_results(samples, pred_masks, num_samples)
    
    avg_loss = total_loss / num_samples
    avg_dice = total_dice / num_samples
    
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")

if __name__ == "__main__":
    main()
