import torch
import matplotlib.pyplot as plt
from train import eval
from dataset import testloader, testset_truth_loader
from torchvision.utils import save_image
from modules import UNet


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning no cuda")



"""Scatter plot for dice scores on test set"""
# Make sure you have a saved model before running this script
x=[]                                                      #List of values for the x-axsis
score_list=[]                                             #List for scores
_,score_list=eval(testloader,testset_truth_loader,"load") #Run a test and fetch the score list

for i in range(len(score_list)):
    x.append(i)
plt.scatter(x,score_list)             #Create scatter plot
plt.show()



"""Creating example images"""
model= UNet()                                               #Define model          
model=model.to(device)                                      #Setting device for model
model.load_state_dict(torch.load('UNet_model.pt'))          #Load the model settings
for image,truth in zip(testloader,testset_truth_loader):
    image = image.to(device)
    truth = truth.to(device)
    prediction = model(image)                                   #Run the image through the model to create predictions

    prediction=torch.round(prediction)
    images = torch.cat((prediction.cpu(),truth.cpu()),dim=0)    #Add the prediction and truth together to create one image
    save_image(images, f'test_img.png',nrow=3)                  #Save a image of the model prediction and the truth. Prediction will be on the left