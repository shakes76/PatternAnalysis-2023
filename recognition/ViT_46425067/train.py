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
    img_channel=1,
    patch_size=16,
    depth=1,
    n_heads=3,
    qkv_bias=True,
    mlp_ratio=4,
)


#load dataloaders
train_loader, test_load, _, _ = load_data(config.batch_size, config.img_size)