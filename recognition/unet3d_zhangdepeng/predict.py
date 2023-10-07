import torch
import numpy as np
import random
import argparse
from modules import UNet3D
from dataset import MRIDataset_pelvis
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
## set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--pth',default='/root/epoch_10_lossmse.pth')

args = parser.parse_args()

model=UNet3D(in_channel=1, out_channel=6).cuda()
model.load_state_dict(torch.load(args.pth))
model.eval()

test_dataset = MRIDataset_pelvis(mode='test',dataset_path='/root/HipMRI_study_complete_release_v1',split_id=200,normalize=True,augmentation=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, f"Shapes don't match {inputs.shape} != {targets.shape}"
        inputs = inputs[:,1:]                                                       # skip background class
        targets = targets[:,1:]                                                     # skip background class
        axes = tuple(range(2, len(inputs.shape)))                                   # sum over elements per sample and per class
        intersection = torch.sum(inputs * targets, axes)
        addition = torch.sum(torch.square(inputs) + torch.square(targets), axes)
        return 1 - torch.mean((2 * intersection + self.smooth) / (addition + self.smooth))
    
criterion = DiceLoss() 

valid_loss=[]
for idx,(data_x,data_y) in enumerate(test_dataloader):
    data_x = data_x.to(torch.float32).cuda()
    data_y = data_y.to(torch.float32).cuda()
    labely=torch.nn.functional.one_hot(data_y.squeeze(1).long(),num_classes=6).permute(0,4,1,2,3).float()
    outputs = model(data_x)
    loss = criterion(outputs,labely)
    valid_loss.append(loss.item())
    print(loss)
m_loss=np.average(valid_loss)
print('average_loss:',m_loss)