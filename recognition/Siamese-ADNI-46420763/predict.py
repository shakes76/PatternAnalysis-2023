"""
predict.py
Example use of the siamese model
 - Samples random image and classifies it
"""
from modules import SiameseNetwork
from dataset import * 

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

BATCH_SIZE = 1
DATASET_DIR = "./recognition/Siamese-ADNI-46420763/data/AD_NC"
MODEL_DIR = "./recognition/Siamese-ADNI-46420763/models/ADNI-SiameseNetwork-sigmoid.pt"

def main():
    ######################    
    #   Retrieve Data:   #
    ######################
    train_dataloader, _, test_dataloader = get_dataloader(DATASET_DIR, BATCH_SIZE, 0.7) 
    
    #########################   
    #   Initialize Model:   #
    #########################
    model = SiameseNetwork()
    model = model.to(device)  
    model.load_state_dict(torch.load(MODEL_DIR, map_location=device))

    #####################   
    #   Get Queries :   #
    #####################
    # Pick random AD and NC query image to test for similarity
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
    
    ###################  
    #   Prediction:   #
    ###################
    model.eval()
    # Get random sample from test:
    image, label = test_dataloader.dataset[random.randint(0, len(test_dataloader.dataset) - 1)]
    image = image[None, :, :, :]
    image = image.to(device)
    
    embedding = model(image)
    AD_query_embedding = model(AD_query)
    NC_query_embedding = model(NC_query)
    
    # Predict similarity for AD and NC
    AD_sim = F.pairwise_distance(AD_query_embedding, embedding, keepdim = True)
    NC_sim = F.pairwise_distance(NC_query_embedding, embedding, keepdim = True)
    
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
    subfigs[0].suptitle("AD Distance: " + "{:.5f}".format(AD_sim.item()), fontsize=20)
    
    # Plot NC Prediction
    axsRight[0].matshow(image[0][0], cmap = plt.cm.binary_r)
    axsRight[0].axis("off")
    axsRight[1].matshow(NC_query[0][0], cmap = plt.cm.binary_r)    
    axsRight[1].axis("off")
    axsRight[0].set_title("Input Image")
    axsRight[1].set_title("NC Query Image")
    subfigs[1].suptitle("NC Distance: " + "{:.5f}".format(NC_sim.item()), fontsize=20)
    
    fig.suptitle("True: " + classes[label] + ", Prediction: " + classes[int(pred)], fontsize=30)
    plt.show()

if __name__ == "__main__":
    main()