#CViT
#GAME PLAN: Convolutional ViT- start with a few convolutional layers to learn hierarchical  eatures from the input images 
#then employ a transformer to handle the higher-level reasoning from the feature maps. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    
    def forward (self, hidden_states):
         # Linear operations on input
        mixed_query_layer = self.query_projection(hidden_states)
        mixed_key_layer = self.key_projection(hidden_states)
        mixed_value_layer = self.value_projection(hidden_states)

        # Transpose for multi-head attention and apply attention mechanism.
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Attention score calculation.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores.mul_(self.scaling)  

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Dropout - help prevent overfitting
        attention_probs = self.dropout(attention_probs)

        # Context layer - weighted sum of the value layer based on attention probabilities.
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # Reshape

        output = self.output_projection(context_layer)  # Projecting back to the original dimension
        return output

class TransformerBlock(nn.Module):
    """
    Transformer Block module comprising of multi-head self-attention mechanism and position-wise feed-forward network (FFN)   
    """
    def __init__(self, config, index):
        super(TransformerBlock, self).__init__()

        # Extracting the configuration parameters based on the block's index
        hidden_size = config.transformer['hidden_size']
        num_heads = config.transformer['num_heads'][index]
        dropout_rate = config.transformer['drop_rate'][index]
        mlp_ratio = config.transformer['mlp_ratios'][index]
        attention_dropout_rate = config.transformer['attention_drop_rate'][index]

        # Ensure the division is integer 
        self.attention_head_size = int(hidden_size // num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        #Layer for MultiHeadSelfAttention
        self.self_attention = MultiHeadSelfAttention(config)
        self.attention_output_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_output_layer_norm = nn.LayerNorm(hidden_size, eps=config.initialisation['layer_norm_eps'])

        # Parameters for the feed-forward network (FFN)
        self.ffn_output_layer_norm = nn.LayerNorm(hidden_size, eps=config.initialisation['layer_norm_eps'])
        ffn_hidden_size = int(hidden_size * mlp_ratio)  # size of the hidden layer in FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),  # GELU activation function
            nn.Dropout(dropout_rate),  # Regularization with dropout
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout_rate),  # Regularization with dropout
        )

        def forward(self, hidden_states):
            # Self-attention part
            attention_output = self.self_attention(hidden_states)
            attention_output = self.attention_output_dropout(attention_output)

            # Adding the residual connection, followed by normalization
            attention_output = self.attention_output_layer_norm(attention_output + hidden_states)

            # Feed-forward network (FFN) part
            ffn_output = self.ffn(attention_output)
            # Adding the residual connection, followed by normalization
            ffn_output = self.ffn_output_layer_norm(ffn_output + attention_output)

            return ffn_output

class ConvolutionalEmbedding(nn.Module):
    """
    Embed images via convolutional layers in CViT- replaces the typical token embedding in a standard transformer model.
    """
    pass

class ConvolutionalVisionTransformer(nn.Module):
    pass


