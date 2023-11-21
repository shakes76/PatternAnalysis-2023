"""
Author: Zach Harbutt S4585714
Shows example usage of trained model
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import modules
import dataset
from torchvision import transforms


root = '/home/groups/comp3710/ADNI/AD_NC'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    
model = modules.ESPCN()
model = model.to(device)
model.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))

test_loader = dataset.ADNIDataLoader(root, mode='test')

toPil = transforms.ToPILImage()

with torch.no_grad():
    model.eval()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))
    for i, (downscaled, orig) in enumerate(test_loader):
        if (i < 2):
            downscaled = downscaled.to(device)
            orig = orig.to(device)
    
            output = model(downscaled)
            
            downscaled = toPil(downscaled.squeeze())
            downscaled = downscaled.resize((256, 240))
            orig = toPil(orig.squeeze())
            output_im = toPil(output.squeeze().cpu().detach())
            output_im.show()
            
            axes[0, i].imshow(np.asarray(downscaled), cmap='gray')
            axes[0, i].set_title('lowres')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(np.asarray(orig), cmap='gray')
            axes[1, i].set_title('highres')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(np.asarray(output.squeeze()), cmap='gray')
            axes[2, i].set_title('reconstruct')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png")