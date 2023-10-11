# containing the source code of the components of your model. 
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn

# Build CNN network and get its embedding vector
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 256*240 -> 128*120

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 128*120 -> 64*60

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 64*60 -> 32*30
            )
        
        self.fc = nn.Sequential(
            nn.Linear(128*32*30, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2),
            )
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


# construct the triplet loss  
class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # calculate euclidean distance from anchor to positive and negative 
        anchor_positive = (anchor - positive).pow(2).sum(1).sqrt()
        anchor_negative = (anchor - negative).pow(2).sum(1).sqrt()

        # print("anchor_positive: ", anchor_positive.mean())
        # print("anchor_negative: ", anchor_negative.mean())

        # calculate loss, use relu to ensure loss are non-negative
        loss = torch.relu(anchor_positive - anchor_negative + self.margin)
        return loss.mean()
    

# construct the siamese network
class SiameseNet(nn.Module):
    def __init__(self, embedding):
        super(SiameseNet, self).__init__()
        self.embedding = embedding

    def forward(self, anchor, positive, negative):
        # use the embedding class to ge the embedding of anchor, positive and negative samples
        embedding_anchor = self.embedding(anchor)
        embedding_positive = self.embedding(positive)
        embedding_negative = self.embedding(negative)
        return embedding_anchor, embedding_positive, embedding_negative
    

# The definition of semi_hard_triplet is the triplets such that the distance of 
# anchor to negative smaller than anchor to positive, but the differences are still smaller than the margin
# This function returns a boolean tensor with shape [batch_size]
def semi_hard_triplet_mining(emb_anchor, emb_positive, emb_negative, margin):
    # calculate distance from anchor to positive and negative samples
    anchor_positive = (emb_anchor - emb_positive).pow(2).sum(1).sqrt()
    anchor_negative = (emb_anchor - emb_negative).pow(2).sum(1).sqrt()

    # print("min posi: ", torch.min(anchor_positive).item())
    # print("max posi: ", torch.max(anchor_positive).item())
    # print("min nega: ", torch.min(anchor_negative).item())
    # print("max nega: ", torch.max(anchor_negative).item())

    # find the semi_hard_triplets
    hard_triplets = anchor_positive > anchor_negative

    semi_hard_triplets = (anchor_positive < anchor_negative) & (anchor_negative - anchor_positive < margin)

    triplet_mask = hard_triplets | semi_hard_triplets
    return triplet_mask