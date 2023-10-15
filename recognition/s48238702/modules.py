import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm, BatchNorm1d

def subnetwork(height, width):
    
    subnet = Sequential(
        Linear(height * width, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
    )

    return subnet

def distance_layer(im1_feature, im2_feature):
   
    tensor = torch.sum(torch.square(im1_feature - im2_feature), dim=1, keepdim=True)
    return torch.sqrt(torch.maximum(tensor, torch.tensor(1e-7)))

def classification_model(subnet):
    pass

def contrastive_loss(y, y_pred):
   
    pass

def siamese(height: int, width: int):
    
    subnet = subnetwork(height, width)

    image1 = torch.nn.Parameter(torch.randn(1, height * width))  # Flatten the input
    image2 = torch.nn.Parameter(torch.randn(1, height * width))  # Flatten the input

    feature1 = subnet(image1)
    feature2 = subnet(image2)

    distance = distance_layer(feature1, feature2)

    # Apply layer normalization to the distance
    tensor = LayerNorm(normalized_shape=1)(distance)

    # Use torch.sigmoid to add the sigmoid activation
    out = torch.sigmoid(torch.nn.Linear(1, 1)(tensor))

    model = torch.nn.Sequential(subnet, out)

    return model