"""
train.py
trains the siamese network
 - validates, tests, and saves trained model
 - plot training performance
"""
from modules import SiameseNetwork
from dataset import * 

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
import torch.nn.functional as F

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#########################    
#   Hyper Parameters:   #
#########################
LOG = True
VISUALIZE = False

# Training Parameters
NUM_EPOCHS = 35
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32

# Model Parameters
MODEL_NAME = "ADNI-SiameseNetwork-sigmoid"
print(MODEL_NAME)
DATASET_DIR = "recognition/Siamese-ADNI-46420763/data/AD_NC"
# DATASET_DIR = "/home/groups/comp3710/ADNI/AD_NC"
LOAD_MODEL = False
MODEL_DIR = None

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(DATASET_DIR, BATCH_SIZE, 0.7)
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork()

    model = model.to(device)  
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    # Criterion and Optimizer:
    distance = CosineSimilarity()
    reducer = ThresholdReducer(high=0.3)
    criterion = losses.TripletMarginLoss(distance = distance, reducer = reducer)
    miner = miners.MultiSimilarityMiner()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)
    
    # Logging
    epoch_valid_acc = []
    epoch_avg_train_loss = []
    epoch_avg_valid_loss = []
    
    ####################
    #   Train Model:   #
    ####################
    total_step = len(train_dataloader)
    print("> Training")
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_correct = 0
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings = model(images)
            # Finds pairs in the batch which are hard to distinguish
            hard_pairs = miner(embeddings, labels) 
            loss = criterion(embeddings, labels, hard_pairs)
                
            # Backward and optimize
            loss.backward()
            optimizer.step()     
            scheduler.step()
            
            total_loss += loss.item()
            
            # Print batch accuracy and loss
            if (i % 100) == 0:
                print("Epoch [{}/{}], Step [{}/{}]  loss: {:.5f}".format(
                    epoch+1, NUM_EPOCHS, 
                    i+1, total_step,
                    loss.item()))
                
        epoch_avg_train_loss.append(total_loss/len(train_dataloader))
            
        #################  
        #   Validate:   #
        #################
        # Get Class Queries
        AD_query, NC_query = get_class_queries(train_dataloader.dataset)
        AD_query_embedding = model(AD_query)
        NC_query_embedding = model(NC_query)
        
        model.eval()
        total_loss = 0.0
        total_correct = 0
        for i, (images, labels) in enumerate(valid_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Calculate Loss
            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs) 
            total_loss += loss.item()
            
            # Calculate Accuracy
            AD_query_batch = AD_query_embedding.repeat(images.shape[0], 1)
            NC_query_batch = NC_query_embedding.repeat(images.shape[0], 1)
            
            ## Predict similarity for AD and NC
            AD_sim = F.pairwise_distance(AD_query_batch, embeddings, keepdim = True)
            NC_sim = F.pairwise_distance(NC_query_batch, embeddings, keepdim = True)
            
            ## Find class with most similarity
            sim = torch.stack([AD_sim, NC_sim], dim = 1)
            pred = torch.Tensor.argmin(sim, dim = 1)

            correct = (labels[:, None] == pred).sum().item()
            total_correct += correct
            
        print("Valid loss: {:.5f}, valid accuracy: {:.2f}%".format(total_loss/len(valid_dataloader), total_correct/len(valid_dataloader.dataset) * 100))
        
        epoch_valid_acc.append(total_correct/len(valid_dataloader.dataset))
        epoch_avg_valid_loss.append(total_loss/len(valid_dataloader))
    
    torch.save(model.state_dict(), "./" + MODEL_NAME + ".pt")
    
    end = time.time()
    elapsed = end - start
    print("Training took ", str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print("END")
    
    ###################
    #   Test Model:   #
    ###################
    model.eval()
    # Get Class Queries
    AD_query, NC_query = get_class_queries(train_dataloader.dataset)
    AD_query_embedding = model(AD_query)
    NC_query_embedding = model(NC_query)
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for i, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings = model(images)
        
        # Calculate Loss
        hard_pairs = miner(embeddings, labels)
        loss = criterion(embeddings, labels, hard_pairs) 
        total_loss += loss.item()
        
        # Calculate Accuracy
        AD_query_batch = AD_query_embedding.repeat(images.shape[0], 1)
        NC_query_batch = NC_query_embedding.repeat(images.shape[0], 1)
        
        ## Predict similarity for AD and NC
        AD_sim = F.pairwise_distance(AD_query_batch, embeddings, keepdim = True)
        NC_sim = F.pairwise_distance(NC_query_batch, embeddings, keepdim = True)
        
        ## Find class with most similarity
        sim = torch.stack([AD_sim, NC_sim], dim = 1)
        pred = torch.Tensor.argmin(sim, dim = 1)

        correct = (labels[:, None] == pred).sum().item()
        total_correct += correct
        
    print("Test loss: {:.5f}, Test accuracy: {:.2f}%".format(total_loss/len(test_dataloader), total_correct/len(test_dataloader.dataset) * 100))
        
    if LOG:
        np.savetxt('epoch_train_loss.csv', epoch_avg_train_loss, delimiter=',')
        np.savetxt('epoch_valid_loss.csv', epoch_avg_valid_loss, delimiter=',')
        np.savetxt('epoch_valid_acc.csv', epoch_valid_acc, delimiter=',')
        
    if VISUALIZE:
        visualize(epoch_valid_acc, epoch_avg_train_loss, epoch_avg_valid_loss)
    
def visualize(epoch_valid_acc, epoch_train_loss, epoch_valid_loss):
    """
    Plots the training performance
    - training loss, validation loss and validation accuracy
    """
    plt.plot(range(len(epoch_train_loss)), epoch_train_loss, label = "Training Loss")
    plt.plot(range(len(epoch_valid_loss)), epoch_valid_loss, label = "Validation Loss")
    plt.grid(True)
    plt.ylabel("Loss")
    plt.xlabel("Epoch #")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    
    plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label = "Validation Accuracy")
    plt.grid(True)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch #")
    plt.title("Validation Accuracy")
    plt.show()

def get_class_queries(dataset):
    """
    Returns a random sample from each class in the dataset
    """
    q_class = -1
    while q_class != 0:
        AD_query, q_class = dataset[random.randint(0, len(dataset) - 1)]
    q_class = -1
    while q_class != 1:
        NC_query, q_class = dataset[random.randint(0, len(dataset) - 1)]
    
    AD_query = AD_query.to(device)
    NC_query = NC_query.to(device)
    AD_query = AD_query[:, None, :, :]
    NC_query = NC_query[:, None, :, :]
    
    return AD_query, NC_query
        
if __name__ == "__main__":
    main()
    
