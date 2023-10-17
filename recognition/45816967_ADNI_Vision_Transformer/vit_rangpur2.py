import torch
from torch.utils.data import DataLoader
from dataset import *
from modules import *
from train import *

# Load datasets and dataloaders
train_set, val_set, test_set = generate_adni_datasets(datasplit=0.1, local=False)

train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
val_loader = DataLoader(val_set, shuffle=True, batch_size=128)
test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

model = ViT(patch_size=16, num_transformer_layers=12, embedding_dim=256, mlp_size=256, num_heads=16).to(device)
summary(model=model, 
				input_size=(128, 1, 224, 224),
				col_names=["input_size", "output_size", "num_params", "trainable"],
				col_width=20,
				row_settings=["var_names"]
	)


train(model, train_loader, val_loader, n_epochs=20, lr = 0.000025, version_prefix="vit11")