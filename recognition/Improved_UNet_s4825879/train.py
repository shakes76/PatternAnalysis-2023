#!/home/terym/miniconda3/envs/pytorch/bin/python3

# import Libraries
import torch
import sys
import matplotlib.pyplot as plt

# import function for image saving
from torchvision.utils import save_image

# import modules and loss function
from modules import ImpUNet, DiceLoss

# import dataloaders
from dataset import train_loader, val_loader 

# Macros
LEARNING_RATE = 5e-4
NUM_EPOCH = 60 

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# variables containing length of dataloaders
total_step = len(train_loader)
total_val_step = len(val_loader)

model = ImpUNet(3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)    
loss_fcn = DiceLoss()

# variable for saving the global best loss
currentBestLoss = float("inf")

# ----------
# TRAINING -
# ----------

for epoch in range(NUM_EPOCH):
    print(f"epoc : {epoch} of {NUM_EPOCH}")
    model.train()
    running_loss = 0.0
    for i, images in enumerate(train_loader):
        img = images[0].to(device)
        truth = images[1].to(device)

        # forward pass
        outputs = model(img).to(device)
        loss = loss_fcn(outputs, truth).to(device)
        
        #backwards and optimize
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item()

        # print status
        if (i + 1) % 15 == 0:
            print("Epoch: [{}/{}], Step: [{}/{}], Loss: {:.5f}"
                  .format(epoch+1, NUM_EPOCH, i+1, total_step, running_loss/(i+1)))
            sys.stdout.flush()
            
            # save image
            outputs = outputs.round()
            saved = torch.cat((outputs, truth), dim=0)
            save_image(saved.view(-1, 1, 64, 64), f"data/prod_img/{epoch}_{i+1}_seg.png", nrow=5)

        # scheduler step
        # scheduler.step()

    # ----------------
    #  EVALUATION    -
    # ----------------

    model.eval()
    total = 0.0
    for i, (img, truth) in enumerate(val_loader):
        # compute without accumulating gradients
        with torch.no_grad():
          # send tensors to device
          img = img.to(device)
          truth = truth.to(device)
          
          # compute outputs
          outputs = model(img)
          
          # compute loss
          loss = loss_fcn(outputs, truth)
          
          # add loss to total loss
          total += loss.item()

          if (i+1) % 10 == 0:
              outputs = outputs.round()
              saved = torch.cat((outputs, truth), dim=0)
              save_image(saved.view(-1, 1, 64, 64), f"data/prod_img/val_{epoch+1}_{i+1}.png")
          
    # Check if new loss is better than best loss
    if total < currentBestLoss:
        print("New best lost calculated... saving model dictionary")
        # upddate the best loss
        currentBestLoss = total
        # save model
        # this ensures that model with best loss is saved
        torch.save(model.state_dict(), "model_dict.pt")         
    
    # print out the total average loss for the validation set
    print("epoch: {}, Loss: {}".format(epoch+1, total/total_val_step))


