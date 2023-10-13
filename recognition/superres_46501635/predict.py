import torch
from modules import ESPCN
from dataset import get_dataloaders
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'C:\\Users\\soonw\\ADNI\\ESPCN_trained_model.pth'  
model = ESPCN(upscale_factor=4)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load a sample from the dataset
_, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC", batch_size=1)
sample_LR, sample_HR, _ = next(iter(test_loader))

# Predict using the model
with torch.no_grad():
    output = model(sample_LR)

# Visualize the results
downsampled_np = sample_LR.numpy()[0][0]
original_np = sample_HR.numpy()[0][0]
output_np = output.numpy()[0][0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(downsampled_np, cmap='gray')
axes[0].set_title('Downsampled Image')
axes[1].imshow(output_np, cmap='gray')
axes[1].set_title('Upsampled by ESPCN')
axes[2].imshow(original_np, cmap='gray')
axes[2].set_title('Original Image')
for ax in axes:
    ax.axis('off')
plt.show()
