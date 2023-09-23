from modules import SiameseNetwork
from dataset import * 

import torch
import torch.nn as nn 
import time
import numpy as np

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#########################    
#   Hyper Parameters:   #
#########################

# Training Parameters
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
BATCH_SIZE = 32

# Model Parameters
MODEL_LAYERS = [1, 32, 64, 128]

MODEL_NAME = "ADNI-SiameseNetwork"
DATASET_DIR = "./recognition/Siamese-ADNI-46420763/data/AD_NC"
LOAD_MODEL = False
MODEL_DIR = None

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    dataloader, valid_dataloader = train_dataloader(DATASET_DIR, BATCH_SIZE, 0.25)
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork(layers=MODEL_LAYERS)
    model = model.to(device)  
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    # Criterion and Optimizer:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    ####################
    #   Train Model:   #
    ####################
    total_step = len(dataloader)
    print("> Training")
    start = time.time()
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_correct = 0
        total_train_loss = 0.0
        for i, (images1, images2, labels) in enumerate(dataloader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred = model(images1, images2)
            loss = criterion(pred, labels[:, None])
            total_train_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            
            pred = torch.Tensor.int(torch.round(pred)) 
            labels = torch.Tensor.int(labels)[:, None]
            
            correct = (pred == labels).sum()
            total_train_correct += correct
            
            # Print batch accuracy and loss
            if (i % 100) == 0:
                print("Epoch [{}/{}], Step [{}/{}]  loss: {:.5f}, acc: {:.5f}".format(
                    epoch+1, NUM_EPOCHS, 
                    i+1, total_step,
                    loss.item(),
                    correct.item()/len(labels)))
        
        train_accuracy = total_train_correct/len(valid_dataloader.dataset)
        
        #######################   
        #   Evaluate Epoch:   #
        #######################
        model.eval()
        total_valid_correct = 0
        total_valid_loss = 0.0
        for i, (images1, images2, labels) in enumerate(valid_dataloader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred = model(images1, images2)
            loss = criterion(pred, labels[:, None])
            total_valid_loss += loss.item()
            
            pred = torch.Tensor.int(torch.round(pred))
            labels = torch.Tensor.int(labels)[:, None]
            
            total_valid_correct += (pred == labels).sum()
            
        valid_accuracy = total_valid_correct/len(valid_dataloader.dataset)
        print("Epoch [{}] Finished: val acc: {:.5f}".format(
                    epoch+1, 
                    valid_accuracy))
    
        epoch_train_loss.append(total_train_loss)
        epoch_valid_loss.append(total_valid_loss)
        epoch_valid_acc.append(valid_accuracy)
        epoch_train_acc.append(train_accuracy)
        
        torch.save(model.state_dict(), "./" + MODEL_NAME + "_EPOCH_5" + ".pt")
        
    np.savetxt('epoch_train_loss.csv', epoch_train_loss, delimiter=',')
    np.savetxt('epoch_valid_loss.csv', epoch_valid_loss, delimiter=',')
    np.savetxt('epoch_valid_acc.csv', epoch_valid_acc, delimiter=',')
    np.savetxt('epoch_train_acc.csv', epoch_train_acc, delimiter=',')
    
    end = time.time()
    elapsed = end - start
    print("Training took ", str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print("END")
    
    ###########################
    #   Save Model Weights:   #
    ###########################
    torch.save(model.state_dict(), "./" + MODEL_NAME + "_EPOCH_15" + ".pt")
    
if __name__ == "__main__":
    main()
    