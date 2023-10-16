import torch
from modules import ImpUNet, DiceLoss
from dataset import ISICDataset
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

BATCH_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((512, 512))])

data = ISICDataset(img_dir="data/train_data", truth_dir="data/train_truth", transform=transform)
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
