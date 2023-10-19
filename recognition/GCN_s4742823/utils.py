'''
This file contains constants used by other classes, and configures the device for rendering on GPU.
'''
import torch

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", str(device))

# This seed is used for random operations in train.py and dataset.py
SEED = 57  # 57 = 91.63% Accuracy
CLASSES = ["Politicians", "Governmental Organisations", "Television Shows", "Companies"]
NUM_CLASSES = len(CLASSES)