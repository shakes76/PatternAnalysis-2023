# import Libraries
import torch
import sys

import matplotlib.pyplot as plt

# import function for image saving
from torchvision.utils import save_image

# import modules and loss function
from modules import model, DiceLoss

# import dataloaders
from dataset import train_loader, val_loader, test_loader, IMAGE_SIZE, BATCH_SIZE

# Macros
LEARNING_RATE = 5*10**(-4)
NUM_EPOCH = 60 

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# variables containing length of dataloaders
total_step = len(train_loader)
total_val_step = len(val_loader)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.985)    
loss_fcn = DiceLoss(smooth=4.0)

# variable for saving the global best loss
currentBestLoss = 0.0 
best_global_min = 0.0

# ----------
# TRAINING -
# ----------
running_loss = 0.0
counter = 0
for epoch in range(NUM_EPOCH):
    print(f"epoch : {epoch + 1} of {NUM_EPOCH}")
    model.train()
    for i, (img, truth) in enumerate(train_loader):
        img = img.to(device)
        truth = truth.to(device)

        # forward pass
        outputs = model(img).to(device)
        loss = loss_fcn(outputs, truth).to(device)
        
        #backwards and optimize
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        # update running loss
        running_loss += loss.item()
        counter += 1

        # print status
        if (i + 1) % 50 == 0:
            print("Epoch: [{}/{}], Step: [{}/{}], Loss: {:.5f}"
                  .format(epoch+1, NUM_EPOCH, i+1, total_step, running_loss/counter))
            sys.stdout.flush()
            
            # save image
            outputs = outputs.round()
            saved = torch.cat((outputs, truth), dim=0)
            save_image(saved.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE), f"data/prod_img/{epoch + 1}_{i+1}_seg.png", nrow=BATCH_SIZE)

        # scheduler step
        scheduler.step()

    # ----------------
    #  EVALUATION    -
    # ----------------

    model.eval()
    total = 0.0
    best_loss = 0.0
    worst_loss = 1.0
    for i, (img, truth) in enumerate(val_loader):
        # compute without accumulating gradients
        with torch.no_grad():
            # send tensors to device
            img = img.to(device)
            truth = truth.to(device)
            
            # compute outputs
            outputs = model(img).round()
            
            # compute loss
            loss = dice(outputs, truth)
            # add loss to total loss
            total += loss.item()

            # save image
            if (i+1) % 50 == 0:
                outputs = outputs.round()
                saved = torch.cat((outputs, truth), dim=0)
                save_image(saved.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE), f"data/prod_img/val_{epoch+1}_{i+1}.png", nrow=BATCH_SIZE)
                    
            # update worst and best loss for evaluation
            if loss.item() < worst_loss:
                worst_loss = loss.item()
            if loss.item() > best_loss:
                best_loss = loss.item()
          
    # Check if new loss is better than best loss
    if total/total_val_step > currentBestLoss:
        print("New best lost calculated... saving model dictionary")
        # upddate the best loss
        currentBestLoss = total/total_val_step
        # save model
        # this ensures that model with best loss is saved
        torch.save(model.state_dict(), "best_average.pt")         

    # check if new parameters create better minimum loss on ecaluation set
    if worst_loss > best_global_min:
        print("new global min found. saving dictionary")
        best_global_min = worst_loss

        torch.save(model.state_dict(), "best_min.pt")
    
    # print out the total average loss for the validation set
    print("epoch: {}, Loss: {}".format(epoch+1, total/total_val_step))
    print("Worst: {:.3f}, Best: {:.3f}".format(worst_loss, best_loss))

#----------
# TESTING -
#----------
# Variable for total dice score
total_dice = 0.0
# create instance of loss function without smooth
loss_test = DiceLoss(smooth=0.0)
# load model dictionary
model.load_state_dict(torch.load('best_min.pt', map_location=torch.device('cpu')))
for i ,(img, truth) in enumerate(test_loader):
    with torch.no_grad():
        img = img.to(device)
        truth = truth.to(device)

        # send image through model and round output for binary segmentation map
        output = model(img).round()

        # compute dice score
        dsc = 1 - loss_test(output, truth).item()
        total_dice += dsc
        
        if (i + 1) % 15 == 0:
            # Print average dice score
            print("\rAverage Loss: {}".format(total_dice/(i + 1)), end="")

