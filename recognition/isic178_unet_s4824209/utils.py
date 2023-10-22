'''
Author: Marius Saether
Student id: 48242099

Mudules used for transformations and loss

Code on lines 75-112 is developed with CHATGPT 3.5, see readme_additions/source_GPT35.txt

'''

import torch.nn as nn
from torchvision import transforms
import torch
import random



class Diceloss(nn.Module):
    '''
    source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch

    Computes the dice loss (1-DCS) of the input and target tensor
    
    args:
        inputs(tensor): tensor to meassure
        targets(tensor): tensor to compare against

    '''

    def __init__(self):
        super(Diceloss, self).__init__()
              
    def forward(self, inputs, targets):    
        
        #flattens input
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #avoids dividing by zero
        eps = 1e-4
        
        #Computes DSC, and returns Dice loss
        intersection = torch.sum(inputs * targets)
        dice = (2.*intersection+eps)/(torch.sum(inputs)+torch.sum(targets)+eps)
        loss = 1-dice
        
        return loss
    


class Train_Transform(object):
    '''
    Transforms applied to the train set
    args:
        data(Tuple): image and corresponding ground truth to transform

    
    Transforms:
        Vertical flip (p=0.25): Vertically flips the input image and the corresponding ground truth  
        
        Horizontal flip (p=0.25): Horizontally flips the input image and the corresponding ground truth
        
        Rotation (p=0.3): Rotates both the input image and ground truth between 1 and 359 degrees 
        **(the rotatonvalue is random, but consistent between image and ground truth)

        ColourJitter(p=0.25): Adjust the brightness, saturation and contrast of input image
        ***This is not applied to the ground truth

        Resize: Resizes the image and ground truth to (size)

        ToTensor: Converts the image and ground truth to tensors

    '''

    def __init__(self, p=0.25, size=(64,64)):
        self.p = p
        self.size = size

    def __call__(self, data):
        image, ground_t = data

        #Vertical flip
        if random.random() < self.p:
            image = transforms.functional.vflip(image)
            ground_t = transforms.functional.vflip(ground_t)

        #Horizontal flip
        if random.random() < self.p:
            image = transforms.functional.hflip(image)
            ground_t = transforms.functional.hflip(ground_t)

        #Rotation
        if random.random() < 0.3:
            random_rotation = random.uniform(1,359)
            image = transforms.functional.rotate(image, angle=random_rotation)
            ground_t = transforms.functional.rotate(ground_t, angle=random_rotation)

        #ColourJitter
        if random.random() < 0.25:
            jitter_transform = transforms.ColorJitter(brightness=(1), contrast=(2.5), saturation=(0.5))
            image = jitter_transform(image)

        
        #Resize 
        image = transforms.functional.resize(image, self.size)
        ground_t = transforms.functional.resize(ground_t, self.size)

        #ToTensor
        image = transforms.functional.to_tensor(image)
        ground_t = transforms.functional.to_tensor(ground_t)

        return image, ground_t




class Test_Transform(object):
    '''
    Aplies transforms.rezise() and transforms.toTensor() to input

    args:
        data(Tuple): image and corresponding ground truth to transform

    Transforms:
        Resize: Resizes image and ground truth to size
        
        ToTensor: Converts the image and ground truth to tensors

    '''

    def __init__(self, size=(64,64)):
        self.size=size

    def __call__(self, data):
        image, ground_t = data

        #Resize 
        image = transforms.functional.resize(image, self.size)
        ground_t = transforms.functional.resize(ground_t, self.size)

        #ToTensor
        image = transforms.functional.to_tensor(image)
        ground_t = transforms.functional.to_tensor(ground_t)

        return image, ground_t
    


