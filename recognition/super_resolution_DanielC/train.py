from modules import *
from utils import *
from dataset import *
import torch
import time

# -------
# Initialise device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

model = SuperResolution().to(device)

# -------
# Generate dataloaders
train_loader = generate_train_loader()
test_loader = generate_test_loader()

# model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

# -------
# Train the model

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lr=learning_rate)

model.train()
print("> Training")
start = time.time()

