"""
Load saved model and show a sample of predictions, alongside the true labels.
"""
import torch
from dataset import ADNIDataset, ADNI_PATH, TEST_TRANSFORM
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import math
import numpy as np


def do_predictions(model, device: torch.device, num_predictions: int = 1, show_plot: bool = True):
    """
    Carry out predictions on the training 
    """
    test_dataset = ADNIDataset(ADNI_PATH, train=False, transform=TEST_TRANSFORM)
    nrc = math.ceil(math.sqrt(num_predictions))
    fig, axes = plt.subplots(nrows=nrc, ncols=nrc, squeeze=False, figsize=(math.ceil(224*nrc/100), math.ceil(224*nrc/100)))
    for i in range(num_predictions):
        # load random image & label
        idx = random.randint(0,test_dataset.__len__()-1)
        image, label = test_dataset.__getitem__(idx)
        image = image.unsqueeze(0).to(device)
        # make prediction
        output = model(image)
        _, predicted = torch.max(output, 1)
        pred = int(predicted[0].item())
        label_strs = {0: "NC", 1: "AD"}
        # show image used for prediction
        image = image.squeeze(0)
        image = image.cpu().numpy()[0,:,:]
        #image.show()
        r = math.floor(i/nrc)
        c = i % nrc
        ax = axes[r,c]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Pred {label_strs[pred]}, real {label_strs[label]}")
        ax.axis('off')

    for i in range(num_predictions, math.ceil(math.sqrt(num_predictions))**2):
        r = math.floor(i/nrc)
        c = i % nrc
        ax = axes[r,c]
        ax.axis('off')

    # Adjust the layout to ensure proper spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # Save/show the plot
    plt.savefig("predictions.png")
    if show_plot:
        plt.show()


def main():
    """Main execution function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not found. Using CPU")
    # Load model
    model = torch.load("adni_vit.pt").to(device)
    do_predictions(model, device, num_predictions=9, show_plot=False)


if __name__ == "__main__":
    main()