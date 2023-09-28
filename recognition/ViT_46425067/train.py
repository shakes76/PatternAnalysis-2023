import torch
import torch.nn as nn
import torch.optim as optim
from modules import ViT
from dataset import load_data
from types import SimpleNamespace

#setup random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
#device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparmeters
config = SimpleNamespace(
    batch_size=32,
    img_size=(224, 224),
    patch_size=16,
    img_channel=1,
    num_classes=2,
    embed_dim=768,
    depth=1,
    num_heads=3,
    mlp_ratio=4,
    qkv_bias=True,
    drop_prob=0.1,
    lr=1e-3
)

#load dataloaders
train_loader, test_load, _, _ = load_data(config.batch_size, config.img_size)

#create model
model = ViT(img_size=config.img_size,
            patch_size=config.patch_size,
            img_channels=config.img_channel,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_prob=config.drop_prob).to(device)

#loss function + optimiser
loss_fn = nn.BCEWithLogitsLoss()
optimiser = optim.AdamW(model.parameters(), lr=config.lr)

