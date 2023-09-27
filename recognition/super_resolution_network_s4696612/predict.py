from dataset import *
from modules import *
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np

path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\AD_NC"
save_path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\PatternAnalysis-2023\recognition\super_resolution_network_s4696612\saved_model.pth"

#-----------------
# Data
batch_size = 64
train_path = path + "\\train\\AD"
test_path = path + "\\test\\AD"
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

path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\AD_NC"
save_path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\PatternAnalysis-2023\recognition\super_resolution_network_s4696612\saved_model.pth"
test_data = ImageDataset(directory=test_path,
                         transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

trained_model = SuperResolution()
trained_model.to(device)
trained_model.eval()
trained_model.load_state_dict(torch.load(save_path))
batch = next(iter(test_loader))
trained_model.eval()
torch.save(trained_model.state_dict(), save_path)

x = batch[0].to(device)[:32]
y = trained_model(x)
a = [y[0], batch[1].to(device)[0]]
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Comparison')
plt.imshow(np.transpose(torchvision.utils.make_grid(a, padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()

plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Model Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(y, padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()

plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Goal Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[:32], padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()