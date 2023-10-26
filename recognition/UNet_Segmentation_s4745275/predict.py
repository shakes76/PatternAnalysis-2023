"""Showing an example useage of the trained model using an image in the images folder. We also compare the prediction to the true mask"""

import os
import matplotlib as plt
from matplotlib import pyplot
from modules import UNet
# import train
from dataset import pre_process_image, pre_process_mask
from PIL import Image
import torch

# If available, its favourable to use the model on a GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example image and the corrsponding mask is located in the images directory
example_path = "recognition/UNet_Segmentation_s4745275/images/ISIC_0000000.jpg"
mask_path = (
    "recognition/UNet_Segmentation_s4745275/images/ISIC_0000000_segmentation.png"
)

# Trained model weights and parameters are stored in best_model.pth
if os.path.exists("recognition/UNet_Segmentation_s4745275/best_model.pth"):
    model_path = "recognition/UNet_Segmentation_s4745275/best_model.pth"
else:
    # Execute the train.py file, which will create the file
    pass #os.system("python train.py")

# Load an instance of the trained model
model = UNet(in_channels=6, num_classes=1)
# model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Load the input image in RGB mode
image = Image.open(example_path).convert("RGB")
# Pre-process the input image
image = (
    pre_process_image(image).unsqueeze(0).to(device)
)  # Add a batch dimension and send to device

with torch.no_grad():
    prediction = model(image)  # This is the prediction of the algorithm
    prediction = (prediction > 0.5).float()  # Binarize the output


# Visual comparison of the predicted segment to the true segment:

# Convert output tensor to numpy array for visualization
predicted_np = prediction.squeeze().cpu().numpy()
fig, ax = pyplot.subplots(1, 2, figsize=(10, 5))

# Open and process the correct mask for visualization
mask = Image.open(mask_path).convert("L")
mask = pre_process_mask(mask)

# True Mask
ax[0].imshow(mask.numpy().transpose(1, 2, 0))
ax[0].set_title("True Mask")

# Predicted Mask
ax[1].imshow(predicted_np, cmap="gray")
ax[1].set_title("Predicted Mask")

plt.show()

