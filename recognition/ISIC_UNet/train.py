import torch
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision.transforms as transforms
from modules import UNet
import torch.optim as optim

##init
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
if not torch.cuda.is_available():
    print("Warning no cuda")

## Data
transform = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor()])

trainset = CustomDataset(root_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2', transform=transform) 
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
segmset = CustomDataset(root_dir='/home/snorrebs/Pattern_3710/ass3_test/ISIC_UNet/pictures_seg', transform=transform) 
segmloader = DataLoader(segmset, batch_size=32, shuffle=False)
total_step=len(trainloader)



def improved_UNet():
    return UNet()

model=improved_UNet()
model=model.to(device)

def dice_loss(model_out,segm_map):
    overlap=(model_out*segm_map).sum()
    dice=(2*overlap)/(model_out.sum()+segm_map.sum())

    return 1-dice


# for images in trainloader:
#     images = images.to(device)
#     output = model(images)
#     image = output[1].cpu()
#     save_image(image, 'prod_img/2.png')
#     print("sover")
#     sleep(100)
optimizer = optim.Adam(model.parameters(), lr=0.0001)  

num_epochs = 10 
model.train()
for epoch in range(num_epochs):
    
    running_loss = 0.0
    i=0
    for images, segm in zip(trainloader,segmloader):  
        images=images.to(device)
        segm=segm.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = dice_loss(outputs, segm)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        #print(images.mean())
        #print("loss")
        #print(loss)
        i=i+1
        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))




    
