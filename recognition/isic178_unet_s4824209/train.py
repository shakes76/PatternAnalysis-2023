from dataset import customDataset
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from modules import IuNet
from utils import Diceloss
from torchvision.utils import save_image
import sys


device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not found, using CPU')
    sys.stdout.flush()

#CONFIG
Num_epochs = 35
LR = 1e-4         
l2_wd = 10e-5       

#LOAD DATA
#path to images and ground truth
d_path = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
g_path = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'

mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]

#convert images and groundtruth to tensors, and resize all images to (512x512)
data_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((256,256))])

GT_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize((256,256))])

#Loading images and corresponding groundtruth as a dataset (each image will have a corresponding groundtruth)
data_set = customDataset(data_path='data', 
                         GT_path='GT_data', 
                         d_transform=data_transform, 
                         g_transform=GT_transform)

#Split dataset into trainset and testset (80/20)
train_lenght = int(0.8*len(data_set))
test_lenght = len(data_set)-train_lenght
train_set, test_set = random_split(data_set, (train_lenght, test_lenght))

#create dataloader for both train and test set
train_loader = DataLoader(train_set, batch_size=1) 
test_loader = DataLoader(test_set, batch_size =1)



#________Model___________
model = IuNet()
model = model.to(device)

criterion = Diceloss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=l2_wd) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.985) 




for epoch in range(Num_epochs):
    running_loss = 0.0
    model.train()
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
    
    scheduler.step()
    img = output.cpu()
    save_image(img, f'segment_img/segment{epoch}.png')

    #test model after each epoch
    model.eval()
    with torch.no_grad():
        avg_DCS = 0.0

        for i, elements in enumerate(test_loader):
            data, ground_t = elements
            output = model(data)
            output = torch.round(output)
            DCS = 1-criterion(output, ground_t)
            avg_DCS += DCS

        print(f'[Test, epoch:{epoch+1}] avg DCS:{avg_DCS/i :.3f}')    

#saving model parameters
torch.save(model.state_dict(), 'trained_model.pt')


