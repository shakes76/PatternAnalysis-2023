import torch
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision.transforms as transforms
from modules import UNet
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

##init
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
if not torch.cuda.is_available():
    print("Warning no cuda")

NUM_EPOCHS = 20
LR=0.00009
SPLIT=0.1


## Data
transform = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor()])


data = CustomDataset(root_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2 ', transform=transform) 
trainset,testset=train_test_split(data,test_size=SPLIT,shuffle=False)
trainloader = DataLoader(trainset, batch_size=6, shuffle=False)
testloader=DataLoader(testset, batch_size=3, shuffle=False)

segmset = CustomDataset(root_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2', transform=transform) 
seg_trainset,seg_testset=train_test_split(segmset,test_size=SPLIT,shuffle=False)
seg_trainloader = DataLoader(seg_trainset, batch_size=6, shuffle=False)
seg_testloader=DataLoader(seg_testset, batch_size=3, shuffle=False)

total_step=len(trainloader)


def improved_UNet():
    return UNet()

model=improved_UNet()
model=model.to(device)

def dice_loss(model_out,segm_map):
    model_out=model_out.view(-1)
    segm_map=segm_map.view(-1)


    overlap=(model_out*segm_map).sum()
    dice=(2*overlap)/(model_out.sum()+segm_map.sum())

    return 1-dice



optimizer = optim.Adam(model.parameters(), lr=LR)  
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=total_step, epochs=5)

#Training
def train():
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        i=0
        for images, segm in zip(trainloader,seg_trainloader):  
            images=images.to(device)
            segm=segm.to(device)
            
            outputs = model(images)
            loss = dice_loss(outputs, segm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            i=i+1

            if (i+1) % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(epoch+1, NUM_EPOCHS, i+1, total_step, running_loss/i))
                #output=torch.round(outputs)
                image = torch.cat((outputs.cpu(),segm.cpu()),dim=0)
                save_image(image, f'better_img/z{epoch}sigm_008_near.png',nrow=6)
        print(epoch)
        test()



#Testing
def test():
    model.eval()
    score_list=[]
    with torch.no_grad():
        for images, truth in zip(testloader, seg_testloader):
            images = images.to(device)
            truth = truth.to(device)

            outputs = model(images)
            outputs=torch.round(outputs)
            for i,x in zip(outputs,truth):
                dice=dice_loss(i,x).item()
                score_list.append(dice)

        print(sum(score_list)/len(score_list))
        print(min(score_list))
        print(max(score_list))
train()






    
