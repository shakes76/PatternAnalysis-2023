"""
    modules.py - model components
"""
import torch
import torch.nn as nn
import torch.nn.functional as Functional
from utils import size_after_cnn_pool

# define the Triplet Loss function
class TripletLoss(nn.Module):
    """Custom loss function for calculating triplet loss."""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, a_out, p_out, n_out):
        """Calculates the triplet loss of the outputs of the anchor, positive and negative.
            L_triplet = max(0, d(a, p) - d(a, n) + m)

        Args:
            a_out (torch.Tensor): network output for the anchor image
            p_out (torch.Tensor): network output for the positive image
            n_out (torch.Tensor): network output for the negative image
        """
        d_ap = Functional.pairwise_distance(a_out, p_out, keepdim=True)
        d_an = Functional.pairwise_distance(a_out, n_out, keepdim=True)
        loss = torch.max(torch.tensor(0), d_ap - d_an + self.margin)
        mean_loss = torch.mean(loss)
        return mean_loss


# define the siamese network
class TripletNetwork(nn.Module):
    """Siamese triplet network architecture."""
    def __init__(self):
        super().__init__()
        cnn_out_1 = 84
        cnn_out_2 = cnn_out_1 * 2
        cnn_out_3 = cnn_out_2 * 2
        cnn_ker_1 = 10
        cnn_ker_2 = 5
        cnn_ker_3 = 3
        cnn_str_1 = 2
        cnn_str_2 = 1
        cnn_str_3 = 1
        self.cnn_layers = nn.Sequential(
            # group 1
            nn.Conv2d(1, cnn_out_1, kernel_size=cnn_ker_1, stride=cnn_str_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.15),

            # group 2
            nn.Conv2d(cnn_out_1, cnn_out_2, kernel_size=cnn_ker_2, stride=cnn_str_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(p=0.15),

            # group 3
            nn.Conv2d(cnn_out_2, cnn_out_3, kernel_size=cnn_ker_3, stride=cnn_str_3),
            nn.ReLU(inplace=True)
            # , nn.MaxPool2d(),
        )
        oh, ow = 256, 240
        l1_h, l1_w = size_after_cnn_pool(oh, ow, cnn_ker_1, cnn_str_1, 2)
        l2_h, l2_w = size_after_cnn_pool(l1_h, l1_w, cnn_ker_2, cnn_str_2, 2)
        l3_h, l3_w = size_after_cnn_pool(l2_h, l2_w, cnn_ker_3, cnn_str_3, 1)

        fc_in_1 = cnn_out_3 * l3_h * l3_w
        fc_out_1 = 2048 # int(fc_in_1 // 16)
        fc_out_2 = int(fc_out_1 // 8)
        fc_out_3 = int(fc_out_2 // 2)
        self.embedding_dim = fc_out_3

        self.fc_layers = nn.Sequential(
            # group 1
            nn.Linear(fc_in_1, fc_out_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),

            # group 2
            nn.Linear(fc_out_1, fc_out_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),

            # final
            nn.Linear(fc_out_2, fc_out_3)
        )

    def single_foward(self, img_tensor):
        """Perform the forward pass for a single image tensor.

        Args:
            img_tensor (_type_): tensor of the image

        Returns:
            fc_output: output of the network.
        """
        # pass the image data through the CNN layers
        cnn_output = self.cnn_layers(img_tensor)
        # flatten the 3D tensor to 2D
        flattened = cnn_output.view(cnn_output.size()[0], -1) 
        # print(img_tensor.size())
        # print(cnn_output.size())
        # print(flattened.size())
        # pass 1D tensor into Fully Connected layers
        fc_output = self.fc_layers(flattened)
        return fc_output
    
    def forward(self, a, p, n):
        """Passes the triplet images through the network.

        Args:
            a (torch.Tensor): anchor image tensor
            p (torch.Tensor): positive image tensor
            n (torch.Tensor): negative image tensor

        Returns:
            a_out (torch.Tensor): network output for the anchor image
            p_out (torch.Tensor): network output for the positive image
            n_out (torch.Tensor): network output for the negative image
        """
        a_out = self.single_foward(a)
        p_out = self.single_foward(p)
        n_out = self.single_foward(n)
        
        return a_out, p_out, n_out
    

# define the classifier
class BinaryClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2),
            nn.Softmax()
        )
    
    def forward(self, input):
        output = self.layers(input)
        return output