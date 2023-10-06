import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MRIDataset_pelvis
import nibabel as nib
# from nibabel import  niftil
# from nibabel.viewers import OrthoSlicer3D

from torch.utils.data import Dataset,DataLoader

class UNet3D(nn.Module):
    ## the out_channel represent the number of division
    def __init__(self, in_channel=1, out_channel=6):
        super(UNet3D, self).__init__()
        ## symmetrical encoder and decoder
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, out_channel, 3, stride=1, padding=1)
        
    def forward(self, x):
        
        #the encoder part
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))

        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
       
        # the decoder part
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        
        return out
    
if __name__=='__main__':
    test_dataset = MRIDataset_pelvis(mode='test',dataset_path='/Users/tongxueqing/Downloads/HipMRI_study_complete_release_v1',split_id=200,normalize=True,augmentation=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    model=UNet3D(in_channel=1, out_channel=6)
    for batch_ndx, sample in enumerate(test_dataloader):
        print('test')
        print(sample[0].shape)
        print(sample[1].shape)
        output=model(sample[0])
        print('output.shape:',output.shape)
        break