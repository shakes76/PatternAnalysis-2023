'''
Author: Marius Saether
student id: 48242099

Program for plotting and evaluating the trained model on the validation set.
'''


from dataset import customDataset, data_sorter
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import IuNet
from utils import Diceloss
from utils import Test_Transform
from torchvision.utils import save_image
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not found, using CPU')
    sys.stdout.flush()


#LOAD DATA
#Root path of validation images and ground truth
img_root = 'validation_data/ISIC2018_Task1-2_Validation_Input'
gt_root = 'validation_data/ISIC2018_Task1_Validation_GroundTruth'

#Creating sorted lists of image and ground truth path for test and train (80%/20% split)
img_test_path,gt_test_path = data_sorter(img_root=img_root, gt_root=gt_root, mode='Test')

#Resize images to (512x512) and convert to tensors
test_transform = transforms.Compose([Test_Transform()])

#Loading the test set into dataloader
test_set = customDataset(images=img_test_path, GT=gt_test_path, transform=test_transform)
#Test loader has batch_size=1 to be able to check dice score of each separate image 
test_loader = DataLoader(test_set, batch_size=1)

#________Model___________
model = IuNet()
model = model.to(device)
trained_model_path = 'trained_model_bestavg.pt'
model.load_state_dict(torch.load(trained_model_path, map_location=device))


criterion = Diceloss()

#test model after each epoch
model.eval()
with torch.no_grad():
    #variable used for computing average Dice coefficient for the entire test set, and max/min score
    avg_DCS = 0.0
    max_DCS = 0
    min_DCS = 1

    DCS_list = []
    y_list = []
    min_DCS_list = []
    #Iterate the test data
    for i, elements in enumerate(test_loader):
        data, ground_t = elements
        data = data.to(device)
        ground_t = ground_t.to(device)
        
        output = model(data)
        
        #The output is rounded so each pixel value <0.5 counts as class 0 and each >0.5 counts as class 1 
        output = torch.round(output)

        #Compute the Dice coefficiet from the dice loss (criterion given by 1-dice)
        DCS = 1-criterion(output, ground_t)
        avg_DCS += DCS.item()

        if DCS.item() > max_DCS:
            max_DCS = DCS.item()

        if DCS.item() < min_DCS:
            min_DCS = DCS.item()
        
        if DCS.item() < 0.1:
            img = torch.cat((torch.round(output), ground_t), dim=0)
            min_DCS_list.append(img)
        
        #Print every 10th image for visualisation
        if i%10 == 9:
            img = torch.cat((torch.round(output), ground_t), dim=0)
            img = img.cpu()
            save_image(img, f'validation_images/batch{i+1}.png', nrow = 1)
            
        #add dcs and index for plotting
        DCS_list.append(DCS.item())
        y_list.append(i+1)
        
    #prints average DCS 
    print(f'[Predict] avg DCS:{avg_DCS/(i+1) :.3f}')


plt.scatter(y_list, DCS_list)
plt.ylabel('DCS score')
plt.savefig('plot/DCS.png')


