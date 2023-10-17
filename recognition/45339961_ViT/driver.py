""" Driver script for creating/loading model, training, and evaluating. """
import config
import torch
from torchinfo import summary
from modules import ViT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    
    
    
    
    if config.will_load:
        model = torch.load(config.load_path).to(device)
    else:
        model = ViT(img_size=config.image_size,
                    in_channels=config.n_channels,
                    patch_size=config.patch_size,
                    num_classes=config.n_classes,
                    num_transformer_layers=config.n_layers,
                    embedding_dim=config.embedding_dim,
                    mlp_size=config.mlp_size,
                    num_heads=config.n_heads,
                    attn_dropout=config.attn_dropout,
                    mlp_dropout=config.mlp_dropout,
                    embedding_dropout=config.embedding_dropout).to(device)

    if config.show_model_summary:
        summary(model=model,
            input_size=(config.batch_size,
                        config.n_channels,
                        config.image_size,
                        config.image_size),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    pass

if __name__ == "__main__":
    main()