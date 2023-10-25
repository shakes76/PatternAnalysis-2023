#CViT
#GAME PLAN: Convolutional ViT- start with a few convolutional layers to learn hierarchical  eatures from the input images 
#then employ a transformer to handle the higher-level reasoning from the feature maps. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

config_params_dict = {
    "general": {
        "num_channels": 3,  # RGB
        "num_classes": 2
    },
    "num_classes": 2,
    "patches": {
        "sizes": [7, 3, 3], # kernel size of each encoder’s patch embedding.
        "strides": [4, 2, 2],  # stride size ^^
        "padding": [2, 1, 1]
    },
    "transformer": {
        "embed_dim": [64, 192, 384],
        "hidden_size": 384, #no. of features in the hidden state 
        "num_heads": [1, 3, 6],  # Matching the number of blocks, balances focus between retaining local feature importance and understanding global context.
        "depth": [1, 1, 1],  # Adjust this according to the number of blocks
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
        self.head_dim = hidden_size // num_heads  # calculate dimension of each head

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by the number of heads ({num_heads})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scaling = self.head_dim ** -0.5  # scaling factor for the dot product attention

        # Setting up the query, key, and value linear projection layers
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        # Define the `all_head_size` attribute
        self.all_head_size = self.head_dim * self.num_heads

        # Output projection layer - takes concatenated output of all attention heads and projects back to the model dimension
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(config.transformer['attention_drop_rate'][0])  # prevent overfitting
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.initialisation['layer_norm_eps'])  # stability

    def transpose_for_scores(self, x):
        """
        Reshapes the 'x' tensor to separate the different attention heads- preparing it for the attention calculation.
        """
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)  # size adjusted based on the number of attention heads and the size of each head.
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # permute to get shape [batch_size, num_heads, seq_length, head_dim]

    def forward(self, hidden_states):
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

        # Dropout - helps prevent overfitting
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
        num_heads = config.transformer['num_heads'][index]  # Corrected to use 'index'
        dropout_rate = config.transformer['drop_rate'][index]
        mlp_ratio = config.transformer['mlp_ratios'][index]
        attention_dropout_rate = config.transformer['attention_drop_rate'][index]

        # Ensure the division is integer
        self.attention_head_size = int(hidden_size // num_heads)
        self.all_head_size = num_heads * self.attention_head_size

        # Layer for MultiHeadSelfAttention
        self.self_attention = MultiHeadSelfAttention(config)  # Pass the 'config' object here
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
    def __init__(self, config):
        super(ConvolutionalEmbedding, self).__init__()

        # Convolutional layers configuration
        self.conv_layers = nn.ModuleList()  # List storing a sequence of convolutions
        self.conv_norms = nn.ModuleList()

        # Extract configuration parameters
        patch_sizes = config.patches['sizes']
        patch_strides = config.patches['strides']
        patch_padding = config.patches['padding']
        embed_dims = config.transformer['embed_dim']

        # Calculate the number of convolutional layers based on the configuration list length
        num_conv_layers = len(patch_sizes)

        for i in range(num_conv_layers):
            # Extract individual configuration parameters for the current layer
            kernel_size = patch_sizes[i]
            stride = patch_strides[i]
            padding = patch_padding[i]
            out_channels = embed_dims[i]

            # Create layer norm for this layer
            layer_norm = nn.LayerNorm(out_channels)

            # Determine the number of input channels for the current layer
            in_channels = config.general['num_channels'] if i == 0 else embed_dims[i - 1]

            # Create the convolutional layer with the current configuration
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

            # Add the created layer and its corresponding layer norm to the module lists
            self.conv_layers.append(conv_layer)
            self.conv_norms.append(layer_norm)

    def forward(self, x):
        # Pass input through convolutional layers
        for conv_layer, layer_norm in zip(self.conv_layers, self.conv_norms):
            x = conv_layer(x)
            x = F.gelu(x)  # Apply GELU activation
            # x = layer_norm(x)  # Apply layer normalization

        # Reshape tensor for compatibility with subsequent transformer layers
        batch_size, embed_dim, height, width = x.size()
        x = x.view(batch_size, embed_dim, -1).transpose(1, 2)  # Flatten spatial dimensions and move embedding dimension

        return x

class ConvolutionalVisionTransformer(nn.Module):
    """
    CViT integrates CNNs with transformers for image processing
    """
    def __init__(self, config):
        super(ConvolutionalVisionTransformer, self).__init__()

        #Initialise convolutional embedding
        self.conv_embedding=ConvolutionalEmbedding(config)

        #Transformer blocks- considering different stages with various depths
        self.transformer_stages = nn.ModuleList()
        block_index = 0 #unified index for all blocks across stages
        for stage_depth in config.transformer['depth']:
            stage_layers=nn.ModuleList()
            for _ in range(stage_depth):
                transformer_block = TransformerBlock(config, block_index)
                stage_layers.append(transformer_block)
                block_index += 1

            self.transformer_stages.append(stage_layers)

        self.final_layer_norm = nn.LayerNorm(config.transformer["hidden_size"], eps=config.initialisation["layer_norm_eps"])
        #classifier head
        self.classifier = nn.Linear(config.transformer['hidden_size'], config.num_classes)

    def forward(self, x):
        #Pass input through convolutional embedding layer
        x = self.conv_embedding(x)

        #Propogate output sequentially through each stage
        for stage in self.transformer_stages:
            for transformer_block in stage:
                x = transformer_block(x)
        x = self.final_layer_norm(x)

        #Flatten representation at token level
        x=x[:, 0]

        #Pass through classification head
        logits = self.classifier(x)
        return logits