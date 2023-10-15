from dataset import customDataset
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import IuNet
from utils import Diceloss
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import sys


device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not found, using CPU')
    sys.stdout.flush()

#CONFIG
Num_epochs = 35
LR = 2e-1           
l2_wd = 10e-5       

#LOAD DATA
d_path = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
g_path = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'

mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]

data_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((192,256)), 
                                transforms.Normalize(mean,std)])

# data_transform = transforms.Compose([transforms.ToTensor(),
#                         transforms.Resize((192,256)),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomVerticalFlip(),
#                         transforms.RandomRotation([-180,180]),
#                         transforms.Normalize(mean=mean,std=std)])

GT_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((192,256))])

train_set = customDataset(data_path='data', GT_path='GT_data', d_transform=data_transform, g_transform=GT_transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True) 




#________Model___________
model = IuNet()
model = model.to(device)

criterion = Diceloss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=l2_wd) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985) 



model.train()

for epoch in range(Num_epochs):
    running_loss = 0.0
    for i, element in enumerate(train_loader):
        
        data, ground_t = element
        data, ground_t = data.to(device), ground_t.to(device)
    
        output = model(data)
        loss = criterion(output, ground_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
        
        if i%10 == 9:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/10 :.3f}')
            sys.stdout.flush()
            running_loss = 0.0

            
            # i = output[0]
            # plt.imshow(i.permute(1,2,0).detach().numpy(), cmap='gray')
            # plt.show()
        
        
    scheduler.step()
    img = output.cpu()
    save_image(img, f'segment_img/segment{epoch}.png')

#saving model parameters
torch.save(model.state_dict(), 'trained_model.pt')
