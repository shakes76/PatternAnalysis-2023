import torch.nn as nn
import torch.nn.functional as F
import torch

# Referred and modified from: https://github.com/shakes76/PatternFlow/blob/topic-recognition/recognition/s4642283-ADNI-SIAMESE/modules.py

####### MODEL 1 ########################

# class EmbeddingNetwork(nn.Module):
#     def __init__(self, input_shape):
#         super(EmbeddingNetwork, self).__init__()

#         #number of channels, number filters, kernel size
#         self.conv1 = nn.Conv2d(input_shape[0], 32, 10)
#         self.conv2 = nn.Conv2d(32, 64, 7)
#         self.conv3 = nn.Conv2d(64, 64, 4)
#         self.conv4 = nn.Conv2d(64, 128, 4)
#         self.fc1 = nn.Linear(128*22*24, 50)# This assumes a specific flattened size from previous layers
#         self.bn = nn.BatchNorm2d(64)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2)
#         x = self.bn(x)
#         x = F.relu(self.conv4(x))
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         return x

# class SiameseNetwork(nn.Module):
#     def __init__(self, input_shape):
#         super(SiameseNetwork, self).__init__()

#         self.embedding_net = EmbeddingNetwork(input_shape)
#         self.fc = nn.Linear(1, 1)

#     def forward_one(self, x):
#         return self.embedding_net(x)

#     def forward(self, x1, x2):
#         output1 = self.forward_one(x1)
#         output2 = self.forward_one(x2)
#         distance = F.pairwise_distance(output1, output2, p=2)
#         out = torch.sigmoid(self.fc(distance.unsqueeze(1)))
#         return out


'''
Alright, let's walk through the shape transformation of the tensor as it passes through the `EmbeddingNetwork`.

1. **Initial Input:** Shape = `[batch_size, channels, 240, 256]`. Assuming grayscale images, channels will be 1.

2. **After `conv1`:** Kernel size = 10 and stride = 1, so:
   Output shape = `[batch_size, 32, 231, 247]` (Because 240-10+1=231 and 256-10+1=247).

3. **After first max pooling:** Pool size = 2:
   Output shape = `[batch_size, 32, 115, 123]`.

4. **After `conv2`:** Kernel size = 7:
   Output shape = `[batch_size, 64, 109, 117]` (Because 115-7+1=109 and 123-7+1=117).

5. **After second max pooling:** Pool size = 2:
   Output shape = `[batch_size, 64, 54, 58]`.

6. **After `conv3`:** Kernel size = 4:
   Output shape = `[batch_size, 64, 51, 55]`.

7. **After third max pooling:** Pool size = 2:
   Output shape = `[batch_size, 64, 25, 27]`.

8. **After `conv4`:** Kernel size = 4:
   Output shape = `[batch_size, 128, 22, 24]`.

'''
####### MODEL 2 ########################

# def contrastive_loss(output1, output2, label, margin=1.0):
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                   (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
#     return loss_contrastive

# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()

#         # Convolutional layers
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, 5),  # Reduced channels
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(32, 64, 5),  # Reduced channels
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2,2)  # Added one more max pooling for dimensionality reduction
#         )
        
#         # Fully connected layers
#         self.fc = nn.Sequential(
#             nn.Linear(64*57*61, 128),  # Adjusted for reduced channels and added pooling
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 2)  # Reduced depth
#         )
        
#     def forward_one(self, x):
#         x = self.conv(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc(x)
#         return x

#     def forward(self, input1, input2):
#         output1 = self.forward_one(input1)
#         output2 = self.forward_one(input2)
#         return output1, output2

'''
In the context of a Siamese Network using contrastive loss, the `margin` represents a hyperparameter that determines how far apart the embeddings of dissimilar items should be. 

The contrastive loss is typically defined as:

\[ \mathcal{L} = (1 - y) \times \text{distance}^2 + y \times \max(0, \text{margin} - \text{distance})^2 \]

Where:
- \( y \) is the label: \( y = 0 \) for similar pairs and \( y = 1 \) for dissimilar pairs.
- \( \text{distance} \) is the distance between the two embeddings (usually Euclidean distance).

Breaking down the formula:
1. For similar pairs (\( y = 0 \)): The loss will be \( \text{distance}^2 \). This means we want the embeddings of similar items to be close to each other.
  
2. For dissimilar pairs (\( y = 1 \)): The loss will be \( \max(0, \text{margin} - \text{distance})^2 \). This encourages the embeddings of dissimilar items to be at least `margin` apart from each other. If the distance is greater than the `margin`, this part of the loss becomes 0, i.e., no penalty.

The `margin` helps ensure that the network doesn't make the embeddings of all items close together to reduce the loss. By forcing the embeddings of dissimilar items to be at least `margin` apart, it adds a penalty when the network fails to do so.

A larger `margin` will force dissimilar embeddings to be further apart. Choosing an appropriate `margin` is crucial, and its value can be determined empirically using a validation set.
'''


## MODEL 3 #########
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Define the subnetwork in the __init__ method
        self.subnetwork = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(128*57*61, 256),  # Adjust the size accordingly
            nn.ReLU(inplace=True),
            #256 representation
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        # Define a layer to transform the distance to a probability
        self.output_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        return self.subnetwork(x)
    
    def forward(self, input1, input2):
        # Compute embeddings for each input
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Compute Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Transform distance to probability
        output = self.output_layer(euclidean_distance)
        
        return output
