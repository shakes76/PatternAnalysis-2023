import torch.nn as nn
import torch.nn.functional as F
import dataset as ds
import torch
import torchvision.models as models


#Resnet Class (50 maybe or 25)
class ADNI_Transformer(nn.Module):
    
    def __init__(self, depth):
        super(ADNI_Transformer, self).__init__()    
        LATENT_DIM = 128
        LATENT_EMB = 64
        
        
        # don't want it pretrianed      
        # take out the classification layer   
        network = models.resnet34(pretrained=False) 
        self._resnet = torch.nn.Sequential(*list(network.children())[:-1])
        
        #initialise the latent array and how many stacks we want
        self.latent = torch.empty(LATENT_DIM, LATENT_EMB)
        self._depth = depth
        
        self._perceiver = nn.ModuleList([Perceiver_Block(LATENT_EMB) for per in range(depth)])
        self._classifier = Classifier()
        

    def forward(self, images):
        # shape 32x3x240x240
        images = self._resnet(images)
        # reshapes to 32x512x1x1
        
        latent = self.latent
        # use perceiver transformer (may n  eed to reshape first)
        for pb in self.perceiver_blocks:
            latent = pb(latent, images)
        # might need to reshape
        
        output = self.classifier(latent)
        return output


class Attention(nn.Module):
    
    def __init__(self, heads, in_size) -> None:
        super(Attention, self).__init__()
        
        self.lnorm1 = nn.LayerNorm(in_size)
        self.attn = nn.MultiheadAttention(embed_dim=in_size, num_heads=heads)

        self.lnorm2 = nn.LayerNorm(in_size)
        self.linear1 = nn.Linear(in_size, in_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(in_size, in_size)
        self.drop = nn.Dropout(0.1)
        
        
        
    def forward(self, in1, in2):
        out = self.lnorm1(in1)
        out, _ = self.attn(query=in1, key=in2, value=in2)
        # out will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] after matmul
        # when used for cross-attention; otherwise same as x
        
        # first residual connection
        resid = out + in2

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out
        
    
    
class MultiAttention(nn.Module):
    def __init__(self, heads, in_size, layers) -> None:
        super(MultiAttention, self).__init__()
        
        self.transformer = nn.ModuleList([
        Attention(
            heads=heads,
            embed_dim=in_size) 
        for layer in range(layers)])
        
    def forward(self, latent, images=None):
        if images == None:
            for head in self.transformer:
                latent = head(latent, latent)
        else:
            for head in self.transformer:
                latent = head(latent, images)
        
        return latent
        
        
    
    
class Classifier(nn.Module):
    
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(128 * 64, 128) 
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(128, 1)
        
        
        
    def forward(self, latent):
        # starts 128x64
        x = self.flatten(latent)
        #reshapes to 128*64x1
        x = self.fc1(x)
        x = self.relu(x)
        #output 128x1
        x = self.fc2(x)
        # output 1x1
        return torch.sigmoid(x)


class Perceiver_Block(nn.Module):
    
    def __init__(self, in_size) -> None:
        super(Perceiver_Block, self).__init__()
        
        self.cross_attention = MultiAttention(1,in_size,1)
        self.latent_transformer = MultiAttention(8,in_size,8)
        
    def forward(self, latent, image):
        l = self.cross_attention(latent, image)

        l = self.latent_transformer(latent)

        return l




