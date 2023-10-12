from dataset import customDataset
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import IuNet

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not found, using CPU')

#CONFIG
Num_epochs = 10
LR = 5e-4

#LOAD DATA
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512))])
train_set = customDataset(data_path='isic178_unet_s4824209/data', transform=transform)
train_loader = DataLoader(train_set, batch_size=10) 


#________Model___________
model = IuNet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(Num_epochs):
    for i, data in enumerate(train_loader):
        
        output = model(data)

