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

BATCH_SIZE = 32
MODEL_LAYERS = [1, 64, 128, 128, 256]
KERNEL_SIZES = [10, 7, 4, 4]
DATASET_DIR = "./recognition/Siamese-ADNI-46420763/data/AD_NC"

MODEL_DIR = "./recognition/Siamese-ADNI-46420763/models/ADNI-SiameseNetwork_EPOCH_15.pt"

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    test_dataloader = get_dataloader(DATASET_DIR, BATCH_SIZE, [0.6, 0.2, 0.2]) 
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork(layers=MODEL_LAYERS, kernel_sizes=KERNEL_SIZES)
    model = model.to(device)  
    model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    ##########################   
    #   Evaluate Test Set:   #
    ##########################
    dataset_targets = test_dataloader.dataset.dataset.targets
    test_indices = test_dataloader.dataset.indices

    # Get the len of each class in the dataset
    num_AD, num_NC = np.bincount([dataset_targets[i] for i in test_indices])
    total_AD_correct = 0
    total_NC_correct = 0

    # Pick random AD and NC query image to test for similarity
    q_class = -1
    while q_class != 0:
        AD_query, q_class = test_dataloader.dataset.dataset[random.choice(test_indices)]
    
    q_class = -1
    while q_class != 1:
        NC_query, q_class = test_dataloader.dataset.dataset[random.choice(test_indices)]
    
    AD_query = AD_query.to(device)
    NC_query = NC_query.to(device)
    AD_query = AD_query[:, None, :, :]
    NC_query = NC_query[:, None, :, :]
    
    print("> Testing")
    start = time.time()
    model.eval()
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Reapeat the queires to match batch size
        AD_query_batch = AD_query.repeat(images.shape[0], 1, 1, 1)
        NC_query_batch = NC_query.repeat(images.shape[0], 1, 1, 1)   
        
        # Predict similarity for AD and NC
        AD_sim = model(images, AD_query_batch)
        NC_sim = model(images, NC_query_batch)
        
        # Find class with most similarity
        sim = torch.stack([AD_sim, NC_sim], dim = 1)
        pred = torch.Tensor.argmin(sim, dim = 1)

        # Get number of correct predictions
        labels = torch.Tensor.int(labels)[:, None]
        correct = (pred == labels)

        # Find number of correct predictions for each class
        AD_mask, NC_mask = (labels == 0), (labels == 1)
        total_AD_correct += (correct * AD_mask).sum()
        total_NC_correct += (correct * NC_mask).sum()
        
    print("Test Accuracy {:.2f} %, AD Accuracy {:.2f} %, NC Accuracy {:.2f} %".format(
            (total_AD_correct + total_NC_correct)/len(test_dataloader.dataset) * 100, 
            (total_AD_correct/num_AD) * 100,
            (total_NC_correct/num_NC) * 100
            )
        )

    print("# AD:", num_AD, "# NC:", num_NC, "# Correct AD:", total_AD_correct, "# Correct AD:", total_NC_correct)

    end = time.time()
    elapsed = end - start
    print("Testing took ", str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print("END")

if __name__ == "__main__":
    main()