#CViT
#GAME PLAN: Convolutional ViT- start with a few convolutional layers to learn hierarchical  eatures from the input images 
#then employ a transformer to handle the higher-level reasoning from the feature maps. 
import torch
import torch.nn as nn

CViT_config_params = {
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
        "num_heads": [1, 3, 6], #balances focus between retaining local feature importance and understanding global context.
        "depth": [1, 2, 10], # relates to overfitting-> adjust in future
        "mlp_ratios": [4.0, 4.0, 4.0, 4.0], # size of the hidden layer: size of the input layer
        "attention_drop_rate": [0.0, 0.0, 0.0], # prevent overfitting
        "drop_rate": [0.0, 0.0, 0.0],
        "drop_path_rate": [0.0, 0.0, 0.1],
        "qkv": {
            "bias": [True, True, True], # queries (q), keys (k), and values (v) 
            "projection_method": ["dw_bn", "dw_bn", "dw_bn"],  # Choose based on your earlier decision
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
    "initialization": {
        "range": 0.02,
        "layer_norm_eps": 1e-6
    }
}


class CViTConfig:
    """
    Configuration class for Convolutional Vision Transformer (CViT) containing the configuration of the 
    CvT- used to instantiate the model with specific architecture parameters.
    """
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

