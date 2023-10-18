"""
Author: Zach Harbutt S4585714
Shows example usage of trained model
"""

import torch
import matplotlib.pyplot as plt
import modules
import dataset
from torchvision import transforms

root = 'AD_NC'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    
model = modules.ESPCN()
model = model.to(device)
model.load_state_dict(torch.load('model.pth'))

test_loader = dataset.ADNIDataLoader(root, mode='test')

toPil = transforms.ToPILImage()

with torch.no_grad():
    fig, axes = plt.subplots(nrows=3, ncols=5)
    for i, (downscaled, orig) in enumerate(test_loader):
        if (i < 5):
            downscaled = downscaled.to(device)
            orig = orig.to(device)
    
            output = model(downscaled)
            
            downscaled = toPil(downscaled[0])
            downscaled = downscaled.resize((256, 240))
            orig = toPil(orig[0])
            output = toPil(output[0])
            
            axes[0, i].imshow(downscaled, cmap='gray')
            axes[0, i].set_title('lowres')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(orig, cmap='gray')
            axes[1, i].set_title('highres')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(output, cmap='gray')
            axes[2, i].set_title('reconstruct')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png")