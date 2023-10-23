import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    #Cross-Attention mechanism where the latent array attends to the input data.
    
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
    # Applies multi-head attention where the query is from latent space and the key-value pairs are from input data.
    
    def forward(self, x, latent):
        # Ensure the sequence length of x and latent match 
        # by truncating or padding latent if necessary
        seq_len_x = x.size(1)
        seq_len_latent = latent.size(1)
        if seq_len_latent < seq_len_x:
            # Pad latent with zeros to match x's sequence length
            padding = torch.zeros(latent.size(0), seq_len_x - seq_len_latent, latent.size(2)).to(latent.device)
            latent = torch.cat([latent, padding], dim=1)
        elif seq_len_latent > seq_len_x:
            # Truncate latent to match x's sequence length
            latent = latent[:, :seq_len_x, :]
        
        # Adjustes dimensions for attention mechanism
        x = x.permute(1, 0, 2)
        latent = latent.permute(1, 0, 2)
        
        # Applying attention
        output, _ = self.attn(latent, x, x)
        output = output.permute(1, 0, 2)
        
        return output + latent


class LatentTransformer(nn.Module):
    #Latent Transformer applies a series of self-attention and feed-forward networks on the latent array.
    
    def __init__(self, embed_dim, num_heads):
        super(LatentTransformer, self).__init__()
        self.self_attention = CrossAttention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    # Feedfoward network 
    def forward(self, latent):
        latent = self.self_attention(latent, latent)
        return self.feedforward(latent)

class Perceiver(nn.Module):
    #The Perceiver model that integrates all components inlcuding embedding layer, crossattention mechanism latent transformer.
    
    def __init__(self, input_dim, latent_dim, embed_dim, n_classes, num_heads):
        super(Perceiver, self).__init__()
    # Embedding layer to tranform
        self.embed = nn.Linear(input_dim, embed_dim)
        self.latent = nn.Parameter(torch.randn(1, latent_dim, embed_dim)) # Initialize latent array with batch dimension
        self.cross_attention = CrossAttention(embed_dim, num_heads)
        self.latent_transformer = LatentTransformer(embed_dim, num_heads)
    # Final classification
        self.classifier = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embed(x)
        x = x.unsqueeze(1)
        
        # Repeat latent for each item in the batch
        latent = self.latent.repeat(x.size(0), 1, 1)
        
        latent = self.cross_attention(x, latent)
        latent = self.latent_transformer(latent)
        latent_mean = latent.mean(dim=1)
        
        return self.classifier(latent_mean)