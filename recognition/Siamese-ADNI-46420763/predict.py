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

BATCH_SIZE = 4
MODEL_LAYERS = [1, 64, 128, 128, 256]
KERNEL_SIZES = [10, 7, 4, 4]
DATASET_DIR = "./recognition/Siamese-ADNI-46420763/data/AD_NC"

MODEL_DIR = "./recognition/Siamese-ADNI-46420763/models/ADNI-SiameseNetwork.pt"

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    train_dataloader, _, test_dataloader = get_dataloader(DATASET_DIR, BATCH_SIZE, [0.6, 0.2, 0.2]) 
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork(layers=MODEL_LAYERS, kernel_sizes=KERNEL_SIZES)
    model = model.to(device)  
    model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    #####################   
    #   Get Queries :   #
    #####################
    test_indices = test_dataloader.dataset.dataset.indices
    train_indices = train_dataloader.dataset.dataset.indices

    # Pick random AD and NC query image to test for similarity
    random.seed(1337) # To maintain the same query across all random test samples
    q_class = -1
    while q_class != 0:
        AD_query, q_class = train_dataloader.dataset.dataset.dataset[random.choice(train_indices)]
    
    q_class = -1
    while q_class != 1:
        NC_query, q_class = train_dataloader.dataset.dataset.dataset[random.choice(train_indices)]
    
    AD_query = AD_query.to(device)
    NC_query = NC_query.to(device)
    AD_query = AD_query[:, None, :, :]
    NC_query = NC_query[:, None, :, :]
    
    ###################  
    #   Prediction:   #
    ###################
    random.seed() # Get random seed
    model.eval()
    # Get random sample from train:
    image, label = train_dataloader.dataset.dataset.dataset[random.choice(test_indices)]
    image = image[None, :, :, :]
    image = image.to(device)
    
    # Predict similarity for AD and NC
    AD_sim = model(image, AD_query)
    NC_sim = model(image, NC_query)
    
    # Find class with most similarity
    sim = torch.stack([AD_sim, NC_sim], dim = 1)
    pred = torch.Tensor.argmin(sim, dim = 1)
    
    classes = ["AD", "NC"]
    print("True:", classes[label])
    print("Pred:", classes[int(pred)])
    
    #################   
    #   Plotting:   #
    #################
    fig = plt.figure(layout='constrained', figsize=(10, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07, hspace=0.0)    
    axsLeft = subfigs[0].subplots(nrows=1, ncols=2)
    axsRight = subfigs[1].subplots(nrows=1, ncols=2)
    
    # Plot AD Prediction
    axsLeft[0].matshow(image[0][0], cmap = plt.cm.binary_r)
    axsLeft[0].axis("off")
    axsLeft[1].matshow(AD_query[0][0], cmap = plt.cm.binary_r)    
    axsLeft[1].axis("off")
    axsLeft[0].set_title("Input Image")
    axsLeft[1].set_title("AD Query Image")
    subfigs[0].suptitle("AD Similarity: " + "{:.5f}".format(AD_sim.item()), fontsize=20)
    
    # Plot NC Prediction
    axsRight[0].matshow(image[0][0], cmap = plt.cm.binary_r)
    axsRight[0].axis("off")
    axsRight[1].matshow(NC_query[0][0], cmap = plt.cm.binary_r)    
    axsRight[1].axis("off")
    axsRight[0].set_title("Input Image")
    axsRight[1].set_title("NC Query Image")
    subfigs[1].suptitle("NC Similarity: " + "{:.5f}".format(NC_sim.item()), fontsize=20)
    
    fig.suptitle("True: " + classes[label] + ", Prediction: " + classes[int(pred)], fontsize=30)
    plt.show()

        
if __name__ == "__main__":
    main()