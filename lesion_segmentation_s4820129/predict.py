import torch
from torch.utils.data import DataLoader
from dataset import TestDataset
import matplotlib.pyplot as plt
import torchvision

#small script that will show how the model will segment a test image
test_dir = '/home/groups/comp3710/ISIC2018/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('path')
model.eval()
dataset = TestDataset(test_dir)
dataloader = DataLoader(TestDataset, batch_size=8)
model.to(device)
for image in dataloader():
    image.to(device)
    out = model
    out_grid_img = torchvision.utils.make_grid(out.cpu(), nrow=4)
    image_grid = torchvision.utils.make_grid(image.cpu(), nrow=4)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(out_grid_img, cmap='gray')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(image_grid)
    plt.savefig('modelpredictions.png')
    plt.show()
    break
    
