import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLO(nn.Module):

    #REFERENCE: yolov3-tiny.cfg from https://github.com/pjreddie/darknet/blob/master/cfg
    #Used as basis for what layers were needed 
    def __init__(self):
        super(YOLO, self).__init__()
        layers = []
        filters = [16,32,64,128,256,512]
        in_channels = 3
        #Convulution layers and maxpooling
        for i in filters:
            layers.append(nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1, bias=False))
            in_channels = i
            layers.append(nn.BatchNorm2d(i))
            layers.append(nn.LeakyReLU(0.1, True)) #might be false
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #Hopefully works
        layers.append(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=1, bias=True))
        self.conv_start = nn.Sequential(*layers)

        #Detection layer - given anchors
        self.anchor1 =  [(81,82), (135,169), (344,319)] #Anchors depends on image?
        
        #Route layer could go here
        self.conv_mid = nn.Sequential(
            nn.Conv2d(255, 128, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode="bilinear"))
        #Another route layer maybe
        self.conv_end = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=1, bias=True))
        
        #Another detection layer
        self.anchor2 = [(10,14), (23,27), (37,58)]

    def forward(self, x):
        out = self.conv_start(x)
        out = out.data
        a = self.predict_transform(out, 416, self.anchor1, 80)
        out = self.conv_mid(out)
        out = self.conv_end(out)
        out = out.data
        b = self.predict_transform(out, 416, self.anchor2, 80)
        return torch.cat((a, b), 1)

    def predict_transform(self, prediction, inp_dim, anchors, num_classes):
        """
        This is everything I need to understand but don't
        """
        batch_size = prediction.size(0)
        stride =  inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)
        
        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
         #Sigmoid the  centre_X, centre_Y. and object confidencce
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
         #Add the center offsets
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

        prediction[:,:,:2] += x_y_offset
        #log space transform height and the width
        anchors = torch.FloatTensor(anchors)
        anchors = anchors.to(device)

        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
        prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
        prediction[:,:,:4] *= stride
        return prediction

        

        