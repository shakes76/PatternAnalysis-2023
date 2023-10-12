# containing the source code of the components of your model. 
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn

# Build CNN network and get its embedding vector
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # convolution layers
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1), # size: 256*240
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 256*240 -> 128*120
            
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 128*120 -> 64*60
            
            # Block 3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 64*60 -> 32*30
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 32*30 -> 16*15

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 16*15 -> 8*7
            nn.AdaptiveMaxPool2d((2, 2)) # size: 8*7 -> 2*2
            )
        
        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 16),
            )

        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
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
# def semi_hard_triplet_mining(emb_anchor, emb_positive, emb_negative, margin):
def semi_hard_triplet_mining(emb_anchor, emb_positive, emb_negative, desire_sample_size, margin):
    # Calculate distance from anchor to positive and negative samples
    anchor_positive = (emb_anchor - emb_positive).pow(2).sum(1).sqrt()
    anchor_negative = (emb_anchor - emb_negative).pow(2).sum(1).sqrt()

    # Find the hard and semi-hard triplets
    hard_triplets = anchor_positive > anchor_negative
    semi_hard_triplets = (anchor_positive < anchor_negative) & (anchor_negative - anchor_positive < margin)

    triplet_mask = hard_triplets | semi_hard_triplets

    if triplet_mask.sum() < desire_sample_size:
        needed_fill_count = desire_sample_size - triplet_mask.sum().item()

        # Compute distance difference for all samples
        semi_hard_distances = anchor_negative - anchor_positive
        semi_hard_distances[~semi_hard_triplets] = float('inf')  # we are only interested in semi-hard samples

        # Retrieve distances for samples that are not yet selected
        unselected_distances = semi_hard_distances[~triplet_mask]
        
        # Sort these distances and get the required number of samples
        _, sorted_indices = torch.sort(unselected_distances, descending=False)
        fill_indices = sorted_indices[:needed_fill_count].tolist()
        
        # Get the global indices for these selected samples
        unselected_samples = torch.where(~triplet_mask)[0]
        global_fill_indices = unselected_samples[fill_indices]

        # Update the triplet mask to include these newly selected samples
        triplet_mask[global_fill_indices] = True

    return triplet_mask
