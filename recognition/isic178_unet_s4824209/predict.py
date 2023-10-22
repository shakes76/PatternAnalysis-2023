'''
Author: Marius Saether
student id: 48242099

Program for plotting av visualising results from a previously trained model.
For program to run: 
insert path to saved model at line 32
insert path to testdata in 142/143 in dataset.py
'''


from dataset import test_loader
from modules import model
import torch 
from utils import Diceloss
from torchvision.utils import save_image
import matplotlib.pyplot as plt


device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not found, using CPU')
    

#________Model___________
trained_model_path = 'trained_model_bestavg.pt'
model = model.to(device)
model.load_state_dict(torch.load(trained_model_path, map_location=device))

#dice loss function, used to calculate DCS (There is no actual loss used here, its just used for DCS calculation)
criterion = Diceloss()


model.eval()
with torch.no_grad():
    #variable used for computing average Dice coefficient for the entire test set, and max/min score
    avg_DCS = 0.0
    max_DCS = 0
    min_DCS = 1

    #Lists used for plotting every DCS value in the test
    DCS_list = []
    y_list = []
    #Iterate the test data
    for i, elements in enumerate(test_loader):
        data, ground_t = elements
        data = data.to(device)
        ground_t = ground_t.to(device)
        
        output = model(data)
        
        #The output is rounded so each pixel value <0.5 counts as class 0 and each >0.5 counts as class 1 
        output = torch.round(output)

        #Compute the Dice coefficiet from the dice loss function (criterion given by 1-dice)
        DCS = 1-criterion(output, ground_t)
        avg_DCS += DCS.item()

        #store lowest DCS score
        if DCS.item() < min_DCS:
            min_DCS = DCS.item()
        
        #Print every 10th image for visualisation
        if i%10 == 9:
            img = torch.cat((torch.round(output), ground_t), dim=0)
            img = img.cpu()
            save_image(img, f'batch{i+1}.png', nrow = 1)
            
        #add dcs and index for plotting
        DCS_list.append(DCS.item())
        y_list.append(i+1)
        
    #prints average DCS 
    print(f'[Predict] avg DCS:{avg_DCS/(i+1) :.3f}')

#Plot DCS results of test
plt.scatter(y_list, DCS_list)
plt.ylabel('DCS score')
plt.savefig('DCS_plot.png')


