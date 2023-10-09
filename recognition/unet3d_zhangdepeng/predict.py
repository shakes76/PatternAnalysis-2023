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
parser.add_argument('--pth',default='/root/epoch_2_lossdice.pth')
parser.add_argument('--dataset_root',default='/root/HipMRI_study_complete_release_v1')

args = parser.parse_args()

##define the model and load trained model file
model=UNet3D(in_channel=1, out_channel=6).cuda()
model.load_state_dict(torch.load(args.pth))
model.eval()

##define the test dataloader
test_dataset = MRIDataset_pelvis(mode='test',dataset_path=args.dataset_root)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


valid_loss=[]
for idx,(data_x,data_y) in enumerate(test_dataloader):
    data_x = data_x.to(torch.float32).cuda()
    data_y = data_y.to(torch.float32).cuda().squeeze()
#     labely=torch.nn.functional.one_hot(data_y.squeeze(1).long(),num_classes=6).permute(0,4,1,2,3).float()

    outputs = model(data_x)
    ##get the class with max value
    outputs_class = torch.argmax(outputs,dim=1).squeeze()
#     print(outputs_class.shape)
    intersection=torch.sum(outputs_class==data_y)
#     print(outputs_class.size(),data_y.size())
    assert outputs_class.size()==data_y.size()
#     print('intersection,outputs_class.size():',intersection,outputs_class.size())
    dice_coeff=intersection.item()/outputs_class.nelement()
    print('dice_coeff',dice_coeff)
    valid_loss.append(dice_coeff)
#     print(loss)
##print the result of test set
m_loss=np.average(valid_loss)
print(args.pth)
print('average_dice_coeff:',m_loss)