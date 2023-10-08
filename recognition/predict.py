import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import ISICDataset, get_transform
from modules2 import UNet2D
from PIL import Image
import numpy as np

# Assuming that the model is saved in a file named 'model.pth'
MODEL_PATH = 'unet2d_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['image'].to(device)
            outputs = model(inputs)
            print("Max and Min of output data:", outputs.max().item(), outputs.min().item())
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            print("Unique values in prediction:", np.unique(preds[0]))
            all_preds.extend(preds)
            # Plotting the first predicted mask
            if i == 0:
                plt.imshow(preds[0], cmap='gray')
                plt.title("Sample Prediction")
                plt.show()
    return all_preds

def visualize_predictions(images, predictions, num_samples=5):
    fig, ax = plt.subplots(num_samples, 2, figsize=(10, 20))
    
    for i in range(num_samples):
        # Option 1: Convert the float32 array to uint8 before converting to a PIL Image.
        img = transforms.ToPILImage()(images[i].squeeze().astype(np.uint8)) 
        
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('Original Image')
        ax[i, 0].axis('off')
        
        ax[i, 1].imshow(predictions[i], cmap='gray')
        ax[i, 1].set_title('Predicted Mask')
        ax[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load model
    model = UNet2D(in_channels=3, num_classes=1).to(DEVICE)  
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Load data
    test_dataset = ISICDataset(dataset_type='test', transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Visualize some images directly from the loader
    for i, batch in enumerate(test_loader):
        images = batch['image'].numpy().transpose((0, 2, 3, 1)) # Bring channel to last axis
        print("Max and Min of image data:", images[0].max(), images[0].min())
        # Assuming your images are in [0, 1] range
        images = images * 255  # Bringing to [0, 255] if they are in [0, 1] range
        fig, axs = plt.subplots(1, 5, figsize=(15, 5))
        for ax, img in zip(axs, images):
            ax.imshow(img.astype(np.uint8))  # Ensure type is uint8
            ax.axis('off')
        plt.show()
        if i == 0: # Breaking after visualizing the first batch
            break
    
    # Get predictions
    predictions = predict(model, test_loader, DEVICE)

    # Visualize
    images = [batch['image'] for batch in test_loader]
    images = torch.cat(images, dim=0).numpy().transpose((0, 2, 3, 1))
    visualize_predictions(images, predictions)
