"""
EXAMPLE USAGE
"""
# Imports
import torch
from dataset import get_test_loader, downscale
from modules import ESPCN
import matplotlib.pyplot as plt

# PyTorch setup
print("PyTorch Version:", torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
model = ESPCN(1, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Get testing data
test_loader = get_test_loader()

# Get input image
input = next(iter(test_loader))[0][0]
input_down = downscale(input)

# Get model prediction
prediction = model(input_down)

# Display results
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
plt.axis("off")
plt.title("Downsampled")
plt.imshow(input_down.permute(1,2,0), cmap='gray')

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Prediction")
plt.imshow(prediction.permute(1,2,0), cmap='gray')

plt.subplot(1,2,3)
plt.axis("off")
plt.title("Original")
plt.imshow(input.permute(1,2,0), cmap='gray')

plt.savefig("prediction.png")