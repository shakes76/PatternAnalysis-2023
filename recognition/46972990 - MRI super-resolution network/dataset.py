import os
import torch

# Check for CUDA availability
print(torch.cuda.is_available())

# Get the name of the GPU (if available)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))