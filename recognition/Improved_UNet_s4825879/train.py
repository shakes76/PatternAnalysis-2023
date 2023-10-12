import torch
from modules import ImpUNet
from dataset import ISICDataset
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 10
LEARNING_RATE = 5*10**(-4)
NUM_EPOCH = 10

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((400, 400))])

data = ISICDataset(img_dir="data", transform=transform)
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

total_step = len(data_loader)

model = ImpUNet(3)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCH):
    for i, img in enumerate(data_loader):
