from dataset import *
from modules import *
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from utils import *


#-----------------
# Data
batch_size = 64

#-------------
# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Warning, CUDA not found. Using CPU.")
print()
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load images for testing
test_data = ImageDataset(directory=test_path,
                         transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# Load saved model
trained_model = SuperResolution()
trained_model.to(device)
trained_model.eval()
trained_model.load_state_dict(torch.load(save_path))
batch = next(iter(test_loader))
trained_model.eval()
torch.save(trained_model.state_dict(), save_path)

# Show images
fig = plt.figure(figsize=(6,3))
x = batch[0].to(device)[:32]
y = trained_model(x)

# Plot a brain, showing the original image, the model generated image, and the goal image
fig.add_subplot(1,3,1)
plt.axis('off')
plt.title('Low Res')
plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[0], padding=2,normalize=True).cpu(), (1,2,0)))

fig.add_subplot(1,3,2)
plt.axis('off')
plt.title('Model Generated')
plt.imshow(np.transpose(torchvision.utils.make_grid(y[0], padding=2,normalize=True).cpu(), (1,2,0)))

fig.add_subplot(1,3,3)
plt.axis('off')
plt.title('High Res')
plt.imshow(np.transpose(torchvision.utils.make_grid(batch[1].to(device)[0], padding=2,normalize=True).cpu(), (1,2,0)))
plt.savefig("Comparison.png")
plt.show()

# Plot a range of model generated images
plt.figure(figsize=(8,4))
plt.axis('off')
plt.title('Model Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(y, padding=2,normalize=True).cpu(), (1,2,0)))
plt.savefig("model_sample.png")
plt.show()

# Plot a range of goal images
plt.figure(figsize=(8,4))
plt.axis('off')
plt.title('Goal Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(batch[1].to(device)[:32], padding=2,normalize=True).cpu(), (1,2,0)))
plt.savefig("goal_images.png")
plt.show()