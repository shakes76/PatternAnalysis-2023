print("Checking if torch is installed correctly with GPU compatitbility: ")
import torch

print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print("Hello world")