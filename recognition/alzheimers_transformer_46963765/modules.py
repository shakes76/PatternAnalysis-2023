import torch.nn as nn
import torch.nn.functional as F
import dataset as ds
import torch
import torchvision.models as models
from torchvision.models import ResNet34_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()
        
        # positional encoding vector 31
        self.position_encodings = torch.randn(embed_dim-1)
        self._convolution = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=1)

        
    def forward(self, images):
        # take the positional encoding vector and expand to the size of the images matrix 32x31x240x240
        enc = self.position_encodings.unsqueeze(0).expand((images.shape[0],) + self.position_encodings.shape).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 240, 240)
        enc = enc.type_as(images) 
       
        # add to the images matrix to make the positional encodings
        images = torch.cat([images, enc], dim=1)
        # flatten the last two dimentions of image to 1d 
        images = images.flatten(2)
        
        # remake permuation of image for attention block
        images = images.permute(2, 0, 1)
        return images


class Attention(nn.Module):
    
    def __init__(self, heads, in_size) -> None:
        super(Attention, self).__init__()
        
        # Cross attention layer as described in paper
        self.lnorm1 = nn.LayerNorm(in_size, device=device)
        self.attn = nn.MultiheadAttention(embed_dim=in_size, num_heads=heads)

        # Dense block as described in the paper. 
        self.lnorm2 = nn.LayerNorm(in_size)
        self.linear1 = nn.Linear(in_size, in_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(in_size, in_size)
        self.drop = nn.Dropout(0.1)


        
    def forward(self, latent, image):
        # in1 is 128 by 64
        #in2 is 32 by 512
                
        out = self.lnorm1(image)
        

        out, _ = self.attn(query=latent, key=image, value=image)
        
        #first residual connection.
        resid = out + latent

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

        # multi head self attention layer as described in paper
        self.transformer = nn.ModuleList([
        Attention(
            heads=heads,
            in_size=in_size) 
        for layer in range(layers)]) 
        self.transformer.to(device=device)
        
    def forward(self, latent):
        
        # self attention so pass latent in twice
        for head in self.transformer:
            latent = head(latent, latent)
        return latent
        
        
class Perceiver_Block(nn.Module):
    
    def __init__(self, in_size, heads, layers) -> None:
        super(Perceiver_Block, self).__init__()
        
        # Perceiver block consists of a single head cross attention
        # and a multi head self attetnion with multiple layers. Was 6 in paper.
        self.cross_attention = Attention(1, in_size)
        self.latent_transformer = MultiAttention(heads,in_size,layers)
        
    def forward(self, latent, image):
        out = self.cross_attention(latent, image)
        out = self.latent_transformer(out)
        return out        
    
    
class Classifier(nn.Module):
    
    def __init__(self, out_dimention, batch_size, latent_size) -> None:
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        
        
    def forward(self, latent):
        
        #32x32x32 latent
        out = self.fc1(latent)
        out = self.relu(out)
        out = out.mean(dim=2).t()
        
        # out 32x1
        return torch.sigmoid(out)



class ADNI_Transformer(nn.Module):
    
    def __init__(self, depth):
        super(ADNI_Transformer, self).__init__()    
        LATENT_DIM = 32
        LATENT_EMB = 32
        latent_layers = 4
        latent_heads = 8
        classifier_out = 128
        batch_size = 32
        
        # pretrained to default values       
        # take out the classification layer   
        
        #network = models.resnet34(weights=ResNet34_Weights.DEFAULT) 
        #self._resnet = torch.nn.Sequential(*list(network.children())[:-1])
        self._embeddings = ImageEncoder(LATENT_EMB)
        
        #initialise the latent array and how many stacks we want
        self.latent = torch.empty(batch_size,LATENT_DIM, LATENT_EMB, device=device)
        self._depth = depth
        
        # Stack perceiver blocks to make final model
        self._perceiver = nn.ModuleList([Perceiver_Block(LATENT_EMB, latent_heads, latent_layers) for per in range(depth)])
        self._perceiver.to(device=device)
        self._classifier = Classifier(classifier_out, batch_size, LATENT_DIM)
        

    def forward(self, images):
        # shape 32x3x240x240
        images = self._embeddings(images)
        
        latent = self.latent
        # latent size 512x512
        
        # use perceiver transformer (may need to reshape first)
        for pb in self._perceiver:
            latent = pb(latent, images)
        
        # classify the output
        output = self._classifier(latent)
        return output



