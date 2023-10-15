import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import torchvision.utils
import numpy as np
CUDA_DEVICE_NUM = 0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    image = image.numpy()   
    plt.imshow(np.transpose(image, (1, 2, 0)), interpolation='nearest')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, batch in enumerate(loader):
        images = batch['image']
        masks = batch['mask']
        x = images.to(device=DEVICE)
        masks = masks.to(DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(masks, f"{folder}{idx}.png")

    model.train()