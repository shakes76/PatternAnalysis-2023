#CViT
#GAME PLAN: Convolutional ViT- start with a few convolutional layers to learn hierarchical  eatures from the input images 
#then employ a transformer to handle the higher-level reasoning from the feature maps. 
import torch
import torch.nn as nn
import torch.nn.functional as F

config_params_dict = {
    "general": {
        "num_channels": 3  # RGB
    },
    "patches": {
        "sizes": [7, 3, 3], # kernel size of each encoder’s patch embedding.
        "strides": [4, 2, 2], # stride size ^^
        "padding": [2, 1, 1]
    },
    "transformer": {
        "embed_dim": [64, 192, 384],
        "hidden_size": 384, #no. of features in the hidden state 
        "num_heads": [1, 3, 6], #balances focus between retaining local feature importance and understanding global context.
        "depth": [1, 2, 10], # relates to overfitting-> adjust in future
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0], # size of the hidden layer: size of the input layer
        "attention_drop_rate": [0.0, 0.0, 0.0], # prevent overfitting
        "drop_rate": [0.0, 0.0, 0.0],
        "drop_path_rate": [0.0, 0.0, 0.1],
        "qkv": {# queries (q), keys (k), and values (v) 
            "bias": [True, True, True], 
            "projection_method": ["dw_bn", "dw_bn", "dw_bn"],  
            "kernel": [3, 3, 3],
            "padding": {
                "kv": [1, 1, 1],
                "q": [1, 1, 1]
            },
            "stride": {
                "kv": [2, 2, 2],
                "q": [1, 1, 1]
            }
        },
        "cls_token": [False, False, True]
    },
    "initialisation": {
        "range": 0.02,
        "layer_norm_eps": 1e-6
    }
}

class CViTConfig:
    """
    Configuration class for Convolutional Vision Transformer (CViT) containing the configuration of the 
    CvT- used to instantiate the model with specific architecture parameters.
    """
    def __init__(self, config_params):
        for key, value in config_params.items():
            setattr(self, key, value)

config = CViTConfig(config_params_dict)

class MultiHeadSelfAttention(nn.Module):
    """
    Implements the multi-head self-attention mechanism. 
    The attention mechanism uses scaled dot-product attention- operates on qkv projection of the input.
    """
    def __init__(self, config):
        super().__init__()
        num_heads = config.transformer['num_heads'][0]  
        hidden_size = config.transformer['hidden_size']
        self.head_dim = hidden_size // num_heads # calculate dimension of each head

        if hidden_size % num_heads != 0: #for future refinement of model 
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by the number of heads ({num_heads})")
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5  # scaling factor for the dot product attention

         # Setting up the query, key, and value linear projection layers
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        # Output projection layer - takes concatenated output of all attention heads and projects back to the model dimension
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(config.transformer['attention_drop_rate'][0]) #prevent overfitting
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.initialisation['layer_norm_eps']) #stability

    def transpose_for_scores(self, x):
        """
        Reshapes the 'x' tensor to separate the different attention heads- preparing it for the attention calculation.
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim) #size adjusted based on number of attention heads and size of each head.
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # permute to get shape [batch_size, num_heads, seq_length, head_dim]
    
    
    def forward ():
        pass



class TransformerBlock(nn.Module):
    pass

class ConvolutionalEmbedding(nn.Module):
    pass

class ConvolutionalVisionTransformer(nn.Module):
    pass


