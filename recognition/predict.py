import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ISICDataset
from modules import ImprovedUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    # converting tensor to numpy array and changing dimensions
    inp = inp.numpy().transpose((1, 2, 0))  
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        # pause a bit so that plots are updated
    plt.pause(0.001)  

def predict():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize your dataset and data loader
    test_dataset = ISICDataset("ISIC2018_Task1-2_Test_Input", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load your model
    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load("model_checkpoint.pth"))
    model.eval()

    # Number of images you want to display
    num_images_to_display = 5

    # Create an iterator for your data loader
    data_iter = iter(test_loader)

    for _ in range(num_images_to_display):
        try:
            images, true_masks = next(data_iter)
        except StopIteration:
            break

        images = images.to(device)
        true_masks = true_masks.to(device)

        with torch.no_grad():
            outputs = model(images)

        # Convert torch tensor to numpy
        predicted_masks = outputs.data.cpu().numpy()
        true_masks = true_masks.cpu().numpy()
        images = images.cpu()

        for image, pred_mask, true_mask in zip(images, predicted_masks, true_masks):
            plt.figure()
            plt.subplot(1, 3, 1)
            imshow(image, title='Original Image')

            plt.subplot(1, 3, 2)
            imshow(np.squeeze(pred_mask), title='Predicted Mask')

            plt.subplot(1, 3, 3)
            imshow(np.squeeze(true_mask), title='True Mask')

            plt.show()

if __name__ == "__main__":
    predict()
