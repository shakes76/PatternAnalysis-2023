# model components
import torch
import torch.nn as nn
import torch.nn.functional as Functional

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


# define the network
class TripletNetwork(nn.Module):
    """Siamese triplet network architecture."""
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # group 1
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # group 2
            nn.Conv2d(6, 12, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # group 3
            nn.Conv2d(12, 24, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
            # , nn.MaxPool2d(),
        )
        self.fc_layers = nn.Sequential(
            # group 1
            nn.Linear(24*59*55, 512),
            nn.ReLU(inplace=True),

            # group 2
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),

            # final
            nn.Linear(128, 2)
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