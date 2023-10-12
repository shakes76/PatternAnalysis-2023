import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

class YOLO(nn.Module):

    #REFERENCE: yolov3-tiny.cfg from https://github.com/pjreddie/darknet/blob/master/cfg
    #Used as basis for what layers were needed 
    def __init__(self):
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
        
        #Route layer?
        self.conv_mid = nn.Sequential(
            nn.Conv2d(255, 128, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True))
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        #Another route layer maybe
        self.conv_end = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=1, bias=True))
        
        #Another detection layer
        self.anchor2 = [(10,14), (23,27), (37,58)]

    


        

        