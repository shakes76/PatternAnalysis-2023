import os
from matplotlib import pyplot as plt
import torch
from dataset import *
import modules
from torch.utils.data import DataLoader
from torch import nn
from torchvision import utils
import matplotlib.pyplot as plt

TEST_PATH = os.path.join(DATA_PATH, 'test', 'ISIC-2017_Test_v2_Data')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = modules.ImprovedUNET(3,1)
model.load_state_dict(torch.load('s4630051_improved_unet_segmentation/save/model_save_final.pth'))

test_dataset = ISICDataset(TEST_PATH, TEST_SEG_PATH, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

model.eval()

model.to(device)

for image in test_loader:
    
    image = image.to(device)
    output = model(image)
        
    out_grid_img = utils.make_grid(output.cpu(), nrow=4)
    image_grid = utils.make_grid(image.cpu(), nrow=4)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    
    ax1.imshow(out_grid_img.permute(1,2,0), cmap='gray')
    ax2 = fig.add_subplot(1,2,2)
    
    ax2.imshow(image_grid.permute(1,2,0))
    plt.savefig('modelpredictions.png')
    
    plt.show()
    break