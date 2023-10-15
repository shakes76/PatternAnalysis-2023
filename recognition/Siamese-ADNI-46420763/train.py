from modules import ResNetEmbedder
from dataset import * 

import torch
import torch.nn as nn 
import time
import numpy as np
import matplotlib.pyplot as plt

from pytorch_metric_learning import miners, losses, distances
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch.nn.functional as F

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#########################    
#   Hyper Parameters:   #
#########################

# Training Parameters
NUM_EPOCHS = 35 # 200
LEARNING_RATE = 0.1 # 0.00005
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 16

# Model Parameters
MODEL_NAME = "ADNI-SiameseNetwork-resnet"
DATASET_DIR = "recognition/Siamese-ADNI-46420763/data/AD_NC"
# DATASET_DIR = "/home/groups/comp3710/ADNI/AD_NC"
LOAD_MODEL = False
MODEL_DIR = None

LOG = True
VISUALIZE = False

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(DATASET_DIR, BATCH_SIZE, 0.7)
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = ResNetEmbedder()

    model = model.to(device)  
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    # Criterion and Optimizer:
    # distance = CosineSimilarity()
    distance = distances.LpDistance()
    criterion = losses.TripletMarginLoss(distance = distance, reducer = ThresholdReducer(high=0.3))
    # miner = miners.MultiSimilarityMiner()
    miner = miners.BatchHardMiner()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)
    
    # Logging Variables
    
    ####################
    #   Train Model:   #
    ####################
    total_step = len(train_dataloader)
    print("> Training")
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_correct = 0
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings = model(images)
            # Finds pairs which are hard to distinguish - the negative sample is sufficiently closs to the anchor
            hard_pairs = miner(embeddings, labels) 
            loss = criterion(embeddings, labels, hard_pairs)

            # Calculate Accuracy:
            positive_indices = hard_pairs[1]
            negative_indices = hard_pairs[2]
            
            pos_sim = F.pairwise_distance(embeddings, embeddings[positive_indices], keepdim = True)
            neg_sim = F.pairwise_distance(embeddings, embeddings[negative_indices], keepdim = True)
            
            # Find class with most similarity
            sim = torch.stack([pos_sim, neg_sim], dim = 1)
            pred = torch.Tensor.argmin(sim, dim = 1)

            correct = (pred == 0).sum().item()
            total_correct += correct
                
            # Backward and optimize
            loss.backward()
            optimizer.step()     
            scheduler.step()
            
            # Print batch accuracy and loss
            if (i % 100) == 0:
                print("Epoch [{}/{}], Step [{}/{}]  loss: {:.5f} train acc: {:.2f}".format(
                    epoch+1, NUM_EPOCHS, 
                    i+1, total_step,
                    loss.item(),
                    correct/len(labels) * 100))
            
        #################  
        #   Validate:   #
        #################
        model.eval()
        
        # Get Class Queries
        q_class = -1
        while q_class != 0:
            AD_query, q_class = train_dataloader.dataset[random.randint(0, len(train_dataloader.dataset) - 1)]
        q_class = -1
        while q_class != 1:
            NC_query, q_class = train_dataloader.dataset[random.randint(0, len(train_dataloader.dataset) - 1)]
        
        AD_query = AD_query.to(device)
        NC_query = NC_query.to(device)
        AD_query = AD_query[:, None, :, :]
        NC_query = NC_query[:, None, :, :]
        AD_query_embedding = model(AD_query)
        NC_query_embedding = model(NC_query)
        
        model.eval()
        # Get random sample from train:
        total_valid_loss = 0.0
        total_correct = 0
        for i, (images, labels) in enumerate(valid_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs) 
            
            total_valid_loss += loss.item()
            
            ####
            
            AD_query_batch = AD_query_embedding.repeat(images.shape[0], 1)
            NC_query_batch = NC_query_embedding.repeat(images.shape[0], 1)
            
            # Predict similarity for AD and NC
            AD_sim = F.pairwise_distance(AD_query_batch, embeddings, keepdim = True)
            NC_sim = F.pairwise_distance(NC_query_batch, embeddings, keepdim = True)
            
            # Find class with most similarity
            sim = torch.stack([AD_sim, NC_sim], dim = 1)
            pred = torch.Tensor.argmin(sim, dim = 1)

            correct = (labels[:, None] == pred).sum().item()
            total_correct += correct
            
        print("Total Valid loss: {:.5f}, accuracy: {:.2f}%".format(total_valid_loss, total_correct/len(valid_dataloader.dataset) * 100))
        
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
    
