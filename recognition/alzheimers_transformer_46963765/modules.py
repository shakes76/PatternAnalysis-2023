"""
Author: Rohan Kollambalath
Student Number: 46963765
COMP3710 Sem2, 2023.
The perceiver model. 
"""

import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    """Method that handles the image encodings for the transformer. Uses a latent array of learned 
       encodings that are appended onto the colour layer. The image is them reshaped to have indexes in order
       2, 0, 1 for the attention layers.     
    """
    def __init__(self, embed_dim):
        """Initialise latent"""
        super(ImageEncoder, self).__init__()
        
        # positional encoding vector to be appended to image
        self.position_encodings = torch.randn(embed_dim-1)
        
    def forward(self, images):
        """Forward pass to apply latent to the second dimention of image"""
        # image is 32x1x240x240
        # take the positional encoding vector 31 and expand to the size of the images matrix 32x63x240x240
        enc = self.position_encodings.unsqueeze(0).expand((images.shape[0],) + self.position_encodings.shape).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 240, 240)
        enc = enc.type_as(images) 
        
        # add to the images matrix to make the positional encoding, image becomes 32x64x240x240
        images = torch.cat([images, enc], dim=1)
        # flatten the last two dimentions of image to 1d, image becomes 32x64x57600
        images = images.flatten(2)
        
        # remake permuation of image for attention block, image becomes 57600x32x64 ready for attention layer
        images = images.permute(2, 0, 1)
        return images


class Attention(nn.Module):
    """Attention layer identical to one mentioned in the paper. Uses
       optional dropout to prevent overfitting to the train set.     
    """
    def __init__(self, heads, in_size) -> None:
        super(Attention, self).__init__()
        """Initialise the respective layers"""
        # Simple attention layer for cross and self attention as described in paper
        # Normalise and pass through attention
        self.lnorm1 = nn.LayerNorm(in_size)
        self.attn = nn.MultiheadAttention(embed_dim=in_size, num_heads=heads)

        # Dense block as described in the paper. 
        self.lnorm2 = nn.LayerNorm(in_size)
        self.linear1 = nn.Linear(in_size, in_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(in_size, in_size)
        # Optional dropout to prevent overfitting
        self.drop = nn.Dropout(0.1)

      
    def forward(self, latent, image):
        """
        Perform the attention. In case of self attention the image will also 
        be a latent array. Residual connections are also used.
        """
        # latent is 32x32x64
        # image is 57600x32x64 in case of self attention image is also latent
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
    """Class that holds multiple self attention layers. This class
       acts as the latent transformer for the model.
    """
    def __init__(self, heads, in_size, layers):
        super(MultiAttention, self).__init__()
        # multi head self attention layer as described in paper
        # multiple layers of multiattention heads
        self.transformer = nn.ModuleList([
        Attention(
            heads=heads,
            in_size=in_size) 
        for layer in range(layers)]) 
        
    def forward(self, latent):
        """Pass through all the self attention layers"""
        # self attention so pass latent in twice
        for head in self.transformer:
            latent = head(latent, latent)
        return latent
        
        
class Perceiver_Block(nn.Module):
    """A block that encapsulates a cross attention layer and a latent transformer"""
    def __init__(self, in_size, heads, layers):
        super(Perceiver_Block, self).__init__()
        """Initialise the two layers"""
        # Perceiver block consists of a single head cross attention
        # and a multi head self attetnion with multiple layers. Was 6 in paper.
        self.cross_attention = Attention(1, in_size)
        self.latent_transformer = MultiAttention(heads,in_size,layers)
        
    def forward(self, latent, image):
        """Forward pass through two layers"""
        # Pass through self and cross attention
        out = self.cross_attention(latent, image)
        out = self.latent_transformer(out)
        return out        
    
    
class Classifier(nn.Module):
    """
    Classifier that recuces latent to binray output
    desired by the BCELoss. 
    """
    def __init__(self, out_dimention):
        super(Classifier, self).__init__()
        """Initialse layers. Lazy layers used as input size is irrelevnant"""
        # Lazy layers don't care about input size. Downsize first and return binary value
        self.fc1 = nn.LazyLinear(out_dimention)
        self.fc2 = nn.LazyLinear(1)
        
    def forward(self, latent):
        """Reduce dimentionality of image before mean and final reduction to binary output"""
        #32x32x64 latent  
        out = self.fc1(latent)
        #32x32x16 latent
        out = out.mean(dim=0)
        #32x16 latent
        out = self.fc2(out)
        
        # 32x1 latent, sigmoid and squeeze to return 31
        return torch.sigmoid(out).squeeze()


class ADNI_Transformer(nn.Module):
    """
    Transformer class that takes in images, does positional encoding and passes through 
    a certain depth of self and cross attention layers (perciever block) and returns a
    binary classification
    """
    def __init__(self, depth, LATENT_DIM, LATENT_EMB, latent_layers, latent_heads, classifier_out, batch_size):
        super(ADNI_Transformer, self).__init__()    
        """
        Defines all layers of the transformer for the specified hyperparameters
        """
        # Hyper Parameters being used
        #LATENT_DIM = 32
        #LATENT_EMB = 64
        #latent_layers = 4
        #latent_heads = 8
        #classifier_out = 16
        #batch_size = 32
        
        # Define image encoder with latent dimetnion 
        self._embeddings = ImageEncoder(LATENT_EMB)
        
        #initialise the latent array and how many stacks we want
        self.latent = torch.empty(batch_size,LATENT_DIM, LATENT_EMB, device=device)
        
        # Stack perceiver blocks to make final model
        self._perceiver = nn.ModuleList([Perceiver_Block(LATENT_EMB, latent_heads, latent_layers) for per in range(depth)])
        self._classifier = Classifier(classifier_out)
        

    def forward(self, images):
        """Forward pass through the entire transformer"""
        # shape of images 32x1x240x240
        images = self._embeddings(images)
        # images becomes 32x32x240x240
        
        # latent size 32x32x64
        latent = self.latent
        
        # use perceiver transformer
        for pb in self._perceiver:
            latent = pb(latent, images)
        
        # classify the output and return
        output = self._classifier(latent)
        return output
    
    
    
    