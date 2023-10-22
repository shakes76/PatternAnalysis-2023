'''
Author: Marius Saether
studetn id: 48242099

Program for training and testing the model

for program to run, correct data paths must be entered at line: 143, 144, 147, 148 in dataset.py

'''
import torch 
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from utils import Diceloss
from modules import model
from dataset import train_loader, validation_loader, test_loader

#Computation will run on GPU if possible 
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not found, using CPU')

#Functions for training, testing and validation
def train(model, train_loader, criterion):
    '''
    Function for training the model
    args:
        model (torch.nn.Module): Model to train
        train_loader (Dataloader): The dataloader with the training set
        criterion(torch.nn.Module): Dice loss function
    '''

    running_loss = 0.0
    model.train()
    
    for i, element in enumerate(train_loader):
        
        #separating train loader into image and ground truth
        image, ground_t = element
        image, ground_t = image.to(device), ground_t.to(device)
        output = model(image)

        #calculation the loss
        loss = criterion(output, ground_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #determening total loss for each epoch   
        running_loss += loss.item()
        
    #prints the avg loss of each epoch
    print(f'[{epoch + 1}] avg_loss: {running_loss/(i+1) :.3f}')
            
    #learning rate scheduler step every epoch
    scheduler.step()

def validate(model, test_loader, criterion):
    '''
    Function for testing the module during training
    args:
        model (nn.Module): Trained model to test
        test_loader (Dataloader): Dataloader with test set
        criterion(nn.Module): Dice loss function

    returns:
        avg_DCS (Float): Average Dice score of the entire test 
        min_dcs (Float): Worst dice score on a single segmentation
    '''

    
    model.eval()
    
    #Variables used to determine average and minimum DCS during testing
    avg_DCS = 0
    min_DCS = 1
    with torch.no_grad():
        #Iterate the validation data
        for i, elements in enumerate(validation_loader):
            data, ground_t = elements
            
            #send to GPU
            data = data.to(device)
            ground_t = ground_t.to(device)
            
            output = model(data)
            
            #The output is rounded so each pixel value <0.5 counts as class 0 and each >0.5 counts as class 1 
            output = torch.round(output)

            #Compute the Dice coefficiet from the dice loss (criterion given by 1-dice)
            DCS = 1-criterion(output, ground_t)
            
            #Determine average DCS and lowest DCS scores
            avg_DCS += DCS.item()
            if DCS.item() < min_DCS:
                min_DCS = DCS.item()
        
        #prints average and minimum DCS 
        print(f'[Validation, epoch:{epoch+1}] avg DCS:{avg_DCS/(i+1) :.3f}, lowest DCS:{round(min_DCS,3)}')

        return avg_DCS, min_DCS 

def test(model, test_loader, criterion):
    '''
    Function for testing the trained model.
    It will load the parameters from the training epoch with best average DCS score on the validation set,
    and plot the DCS scores of all images in the test-set. 

    '''
    
    #Loading parameters with best average score from training
    model.load_state_dict(torch.load('trained_model_bestavg.pt'))
    model.eval()
    with torch.no_grad():
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

            #Store lowest DCS score of any image
            if DCS.item() < min_DCS:
                min_DCS = DCS.item()
            
            #variables used for plotting
            DCS_list.append(DCS.item())
            y_list.append(i+1)

        print(f'[Predict] avg DCS:{avg_DCS/(i+1) :.3f}, lowest DCS:{round(min_DCS,3)}')

        #Scatterplot of all DCS values of validation set
        plt.scatter(y_list, DCS_list)
        plt.ylabel('DCS score')
        plt.savefig('test_DCS_plot.png')

#PARAMETERS
Num_epochs = 3
batch_size = 2 #will only be applied to the trainset
LR = 5e-4         

#MODEL
model = model.to(device)

#create a Dice loss function, Adam optimizer, and a step learning rate scheduler
criterion = Diceloss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.985) 


#TRAINING / VALIDATION
#comparison variables to save the models with best min/avg dcs score
best_min_dcs = 0
best_avg_dcs = 0
for epoch in range(Num_epochs):
    #train model
    print('>>>training')
    train(model, train_loader, criterion)
    
    #validate model
    print('>>>validating')
    avg_dcs, min_dcs = validate(model, validation_loader, criterion)

    #save model with best minimum DCS score:
    if min_dcs > best_min_dcs:
        torch.save(model.state_dict(), 'trained_model_bestmin.pt')
        best_min_dcs = min_dcs
    
    #save model with best average DCS score:
    if avg_dcs > best_avg_dcs:
        torch.save(model.state_dict(), 'trained_model_bestavg.pt')
        best_avg_dcs = avg_dcs

#TESTING
print('<<<testing')
test(model, test_loader, criterion)

