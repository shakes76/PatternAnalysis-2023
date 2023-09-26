import torch
import torchvision

class Transformer(torch.nn.Module):

    def __init__(self, nPatches, patchSize):
        super().__init__()
        self.linEmbed = torch.nn.Linear(patchSize*patchSize, patchSize*patchSize)
        self.nPatches = nPatches
        return None
    
    def forward(self, imagePatches):

        imagePatches = torch.flatten(imagePatches, start_dim=3) #flattens the 2d image data
        imagePatches = torch.flatten(imagePatches, start_dim=1, end_dim=2) #lines the patches up along a single dimension

        #linearly embed each of the image patches
        for i in range(self.nPatches):
            self.linEmbed(imagePatches[:, i, :])
        
        #linearly embed the images

        

