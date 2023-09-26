from modules import SiameseNetwork
from dataset import * 

import torch
import torch.nn as nn 
import time
import numpy as np
import matplotlib.pyplot as plt

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#########################    
#   Hyper Parameters:   #
#########################

# Training Parameters
NUM_EPOCHS = 30
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 32

# Model Parameters
MODEL_LAYERS = [1, 64, 128, 128, 256]
KERNEL_SIZES = [10, 7, 4, 4]

MODEL_NAME = "ADNI-SiameseNetwork"
DATASET_DIR = "recognition/Siamese-ADNI-46420763/data/AD_NC"
LOAD_MODEL = False
MODEL_DIR = None

LOG = True
VISUALIZE = False

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    train_dataloader, valid_dataloader, _ = get_dataloader(DATASET_DIR, BATCH_SIZE, [0.6, 0.2, 0.2])
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork(layers=MODEL_LAYERS, kernel_sizes=KERNEL_SIZES)

    model = model.to(device)  
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    # Criterion and Optimizer:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    ####################
    #   Train Model:   #
    ####################
    total_step = len(train_dataloader)
    print("> Training")
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_correct = 0
        total_train_loss = 0.0
        for i, (images1, images2, labels) in enumerate(train_dataloader):
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
            
            # Predict similarity
            pred = torch.Tensor.int(torch.round(pred)) 
            labels = torch.Tensor.int(labels)[:, None]
            
            # Evaluate prediction
            correct = (pred == labels).sum()
            total_train_correct += correct.item()
            
            # Print batch accuracy and loss
            if (i % 100) == 0:
                print("Epoch [{}/{}], Step [{}/{}]  loss: {:.5f} acc: {:.5f}".format(
                    epoch+1, NUM_EPOCHS, 
                    i+1, total_step,
                    loss.item(),
                    correct.item()/len(labels)))
        
        train_accuracy = total_train_correct/len(train_dataloader.dataset)
        
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
            
            total_valid_correct += (pred == labels).sum().item()
            
        valid_accuracy = total_valid_correct/len(valid_dataloader.dataset)
        
        print("Epoch [{}] validition loss: {:.5f} validation acc: {:.5f}".format(
                    epoch+1, 
                    total_valid_loss/len(valid_dataloader),
                    valid_accuracy
                    ))
    
        epoch_train_loss.append(total_train_loss/len(train_dataloader))
        epoch_valid_loss.append(total_valid_loss/len(valid_dataloader))
        epoch_valid_acc.append(valid_accuracy)
        epoch_train_acc.append(train_accuracy)

    if LOG == True:
        np.savetxt('epoch_train_loss.csv', epoch_train_loss, delimiter=',')
        np.savetxt('epoch_valid_loss.csv', epoch_valid_loss, delimiter=',')
        np.savetxt('epoch_valid_acc.csv', epoch_valid_acc, delimiter=',')
        np.savetxt('epoch_train_acc.csv', epoch_train_acc, delimiter=',')
    
    if VISUALIZE == True:
        visualize(epoch_train_acc, epoch_valid_acc, epoch_train_loss, epoch_valid_loss)
    
    end = time.time()
    elapsed = end - start
    print("Training took ", str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print("END")
    
    ###########################
    #   Save Model Weights:   #
    ###########################
    torch.save(model.state_dict(), "./" + MODEL_NAME + ".pt")
    
    
def visualize(epoch_train_acc, epoch_valid_acc, epoch_train_loss, epoch_valid_loss):
    plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label = "Training Accuracy")
    plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label = "Validation Accuracy")
    plt.plot(range(len(epoch_train_loss)), epoch_train_loss, label = "Training Loss")
    plt.plot(range(len(epoch_valid_loss)), epoch_valid_loss, label = "Validation Loss")
    plt.grid(True)
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epoch #")
    plt.title("Training Loss and Accuracy")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
    
