import torch
from torch.utils.data import DataLoader
from dataset import *
from modules import *
from train import *
from predict import *

# Create datasets and dataloaders
train_set, val_set, test_set = generate_adni_datasets(datasplit=0.1)

train_loader = DataLoader(train_set, shuffle=True, batch_size=64)
val_loader = DataLoader(val_set, shuffle=True, batch_size=64)
test_loader = DataLoader(test_set, shuffle=False, batch_size=64)

# Create model
model = ViT(patch_size=16, num_transformer_layers=12, embedding_dim=256, mlp_size=256, num_heads=16).to(device)
# Train model
train(model, train_loader, val_loader, n_epochs=50, lr = 0.000015, version_prefix="vit22")
# Test model
test(model, test_loader)
# Run predictions using model
predict(model, test_loader, (2,5), "driver_vit")