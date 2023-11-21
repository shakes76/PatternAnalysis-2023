import torch
from modules import ImpUNet, DiceLoss
from dataset import test_loader 
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import torch.nn as nn
import os

import matplotlib.pyplot as plt

# True for plotting images, groundtruth and segmentation
PLOT_IMAGES = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_step = len(test_loader)

model = ImpUNet(3).to(device)
model.load_state_dict(torch.load('best_min.pt', map_location=torch.device('cpu')))
loss_fcn = DiceLoss(smooth=0)


total = 0.0
lossList = []
model.eval()
for i, (image, truth) in enumerate(test_loader):
    with torch.no_grad():
        img = image.to(device)
        truth = truth.to(device)

        output = model(img).round()

        loss = 1 - loss_fcn(output, truth).item()

        total += loss
        lossList.append(loss)
        
        if PLOT_IMAGES:
            image = img[0,:,:,:]
            image = torch.squeeze(image, dim=0)
            segmentation = output[0,0,:,:].round()
            truth = truth[0,0,:,:]
            
            toPil = ToPILImage()
            image = toPil(image)
            
            # create figure with two subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))
            
            ax1.set_title('Image')
            ax1.imshow(image)
            
            ax2.set_title('segmentation map')
            ax2.imshow(segmentation.detach().numpy(), cmap='gray')

            ax3.set_title('truth')
            ax3.imshow(truth.detach().numpy(), cmap='gray')
       
            plt.show()

plt.scatter(range(len(lossList)), lossList)
plt.show()

print("total: {}".format(total/total_step))

