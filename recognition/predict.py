import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import ISICDataset, get_transform, get_mask_transform  # Import the necessary functions
from modules import ImprovedUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Predict: {torch.cuda.is_available()}")

def imshow(inp, title=None, gray=False, ax=None):
    """Imshow for Tensor."""
    # Convert tensor to numpy if it's a tensor
    if isinstance(inp, torch.Tensor):
        inp = inp.numpy()
    if inp.shape[0] == 1:
        inp = np.squeeze(inp, axis=0)
    if gray:
        cmap = 'gray'
    else:
        cmap = None
    if len(inp.shape) == 3:
        inp = inp.transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    if ax is None:
        plt.imshow(inp, cmap=cmap)
        if title is not None:
            plt.title(title)
    else:
        ax.imshow(inp, cmap=cmap)
        ax.set_title(title)
    plt.pause(0.001)



# Custom collate function to handle None masks
def my_collate(batch):
    # Filter out entries where the mask is None
    new_batch = list(filter(lambda x: x[1] is not None, batch))

    # Check if there are any valid samples in the batch
    if len(new_batch) == 0:  # if all masks were None, we skip this batch
        print("Warning: Empty batch encountered. All masks were None.")
        return None  # You might want to return an identifiable value here for easier handling later.

    # If there's at least one valid sample, we create a batch for processing
    return torch.utils.data.dataloader.default_collate(new_batch)



def predict():
    img_transform = get_transform()
    mask_transform = get_mask_transform()

    test_dataset = ISICDataset(image_dir="ISIC2018_Task1-2_Training_Input_x2",
                               mask_dir="ISIC2018_Task1_Training_GroundTruth_x2",
                               img_transform=img_transform,
                               mask_transform=mask_transform,
                               img_size=(1024, 1024))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate)

    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load("plot_checkpoint.pth"))
    model.eval()

    num_images_to_display = 5

    data_iter = iter(test_loader)

    for _ in range(num_images_to_display):
        try:
            batch = next(data_iter)
            if batch is None:
                print("Empty batch encountered; all masks were None for this iteration.")
                continue  # Skip this loop iteration if the batch is empty

            images, true_masks = batch  # Unpack your non-empty batch here
        except StopIteration:
            break

        images = images.to(device)
        true_masks = true_masks.to(device)

        with torch.no_grad():
            outputs = model(images)

        predicted_masks = outputs.data.cpu().numpy()
        true_masks = true_masks.cpu().numpy()
        images = images.cpu()

        for image, pred_mask, true_mask in zip(images, predicted_masks, true_masks):
            plt.subplot(1, 3, 1)
            imshow(image, title='Original Image')

            plt.subplot(1, 3, 2)
            imshow(np.squeeze(pred_mask), title='Predicted Mask', gray=True)

            plt.subplot(1, 3, 3)
            imshow(np.squeeze(true_mask), title='True Mask', gray=True)

# Main block
if __name__ == "__main__":
    predict()
    plt.show(block=True)
