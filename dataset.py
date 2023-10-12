import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import modules as m

train_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
seg_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),]
)

transform_test = transforms.Compose(
	  [transforms.ToTensor(),
  	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

train_loader = m.ISICDataLoader(train_path, seg_path, transform_train, transform_test)

print(train_loader.__getitem__(1))