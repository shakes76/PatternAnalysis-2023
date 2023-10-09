import torch
import matplotlib.pyplot as plt
from modules import SuperResolutionModel
from dataset import ADNIDataset

# Initialize the dataset and data loaders
data_loader = ADNIDataset()
test_loader = data_loader.test_loader

# Initialize the model and load trained weights
model = SuperResolutionModel()
model.load_state_dict(torch.load("super_resolution_model.pth"))
model.eval()

# Predict and visualize super-resolved images
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        outputs = model(inputs)

        # Visualize input, target, and output images
        for i in range(len(inputs)):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title("Input")
            plt.imshow(inputs[i][0].cpu().numpy(), cmap="gray")

            plt.subplot(1, 3, 2)
            plt.title("Target (High-Resolution)")
            plt.imshow(targets[i][0].cpu().numpy(), cmap="gray")

            plt.subplot(1, 3, 3)
            plt.title("Super-Resolved Output")
            plt.imshow(outputs[i][0].cpu().numpy(), cmap="gray")

            plt.show()