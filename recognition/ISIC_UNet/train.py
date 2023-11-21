import torch
from modules import UNet
from torchvision.utils import save_image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils import dice_loss
from dataset import trainloader, validationloader, truth_validationset_loader, testset_truth_loader, testloader


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')     #Trying to define device as GPU
if not torch.cuda.is_available():                                       #Print warning if GPU is not available
    print("Warning no cuda")

NUM_EPOCHS = 2                   #Defining number of epochs
LR=0.00009                       #Defining learning rate

total_step=len(trainloader)      #Define lenght of batched dataset

#Innit model
def improved_UNet():
    return UNet()

model=improved_UNet()  
model=model.to(device)           #Set model to run on GPU

optimizer = optim.Adam(model.parameters(), lr=LR)               #Adam optimizer for loss calculation
scheduler = StepLR(optimizer, step_size=total_step, gamma=0.95) # SetpLR for LR-scheduler


#Training
def train():
    score=0                        #For evaluating performance of models
    new_score=0                                     
    for epoch in range(NUM_EPOCHS):
        print("---------Training---------")
        model.train()                               #Set the model for trainings
        running_loss = 0.0
        i=0
        for images, segm in trainloader:            #Run through the batches in the data loader
            images=images.to(device)                #Set the device to GPU
            segm=segm.to(device)
            
            prediction = model(images)              #Run the images through the model to get a prediction
            loss = dice_loss(prediction, segm)      #Calculate loss    

            optimizer.zero_grad()                   #Clears out the gradient parameters for the model
            loss.backward()                         #Adjusts prameters for the model
            optimizer.step()                        #Progress the optimizer
            scheduler.step()                        #Progress the scheduler
            running_loss += loss.item()             #Calculate runningloss
            i=i+1

            if (i+1) % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(epoch+1, NUM_EPOCHS, i+1, total_step, running_loss/(i+1)))
    
        print("------------------------------")
        print(f"Performing validation for epoch number {epoch+1}:")
        new_score,_=eval(validationloader,truth_validationset_loader,"no_load")  #Run evaluation

        if new_score>=score:                                        #Checking if new score is bettern than old score
            score=new_score                                         #Define new best score
            print("Saving model ...")                   
            torch.save(model.state_dict(), 'UNet_model.pt')         #Saving the model



#Validating and testing
def eval(input,truth,load):
    model.eval()                                            #Innit the model for eval
    if load == "load":                                      #Load the best model when testing
        print("Loading model ...")
        model.load_state_dict(torch.load('UNet_model.pt'))  #Loading model
    score_list=[]
    with torch.no_grad():                                   
        for images, truth_batch in zip(input,truth):        #Loading batches
            images = images.to(device)
            truth_batch = truth_batch.to(device)            #Setting device for truth and images

            outputs = model(images)
            outputs=torch.round(outputs)                    #Rounding the predictions from the model to get binary values
            for i,x in zip(outputs,truth_batch):            #Calculate dice score for all the images in the batch
                dice=1-dice_loss(i,x).item()
                score_list.append(dice)                     #Apennd to the list for all scores in test set.

        print(f"Mean: {sum(score_list)/len(score_list)}")
        print(f"Lowest score: {min(score_list)}")
        print(f"Highest score: {max(score_list)}")
        return sum(score_list)/len(score_list),score_list  #Return the mean and the score list
    
if __name__=="__main__":
    train()
    print("-----------------------")
    print("Performing test:")
    eval(testloader,testset_truth_loader,"load")






    
