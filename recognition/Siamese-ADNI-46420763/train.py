from modules import SiameseNetwork
from dataset import * 
from utils import ContrastiveLoss

import torch
import torch.nn as nn 
import time

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#########################    
#   Hyper Parameters:   #
#########################

# Training Parameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
BATCH_SIZE = 64

# Model Parameters
MODEL_LAYERS = [1, 32, 64, 128, 256]

MODEL_NAME = "ADNI-SiameseNetwork"
DATASET_DIR = "./recognition/Siamese-ADNI-46420763/data/AD_NC"
LOAD_MODEL = False
MODEL_DIR = None

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    dataloader = train_dataloader(DATASET_DIR, BATCH_SIZE)
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork(layers=MODEL_LAYERS)
    model = model.to(device)  
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    # Criterion and Optimizer:
    criterion = ContrastiveLoss()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    
    ####################
    #   Train Model:   #
    ####################
    total_step = len(dataloader)
    print("> Training")
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (images1, images2, labels) in enumerate(dataloader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output1, output2, pred = model(images1, images2)
           
            loss = criterion(output1, output2, labels) # Contrastive Loss
            # loss = criterion(pred, labels[:, None]) # BCE LOSS
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
       
if __name__ == "__main__":
    main()
    