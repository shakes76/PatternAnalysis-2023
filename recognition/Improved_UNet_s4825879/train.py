import torch
from modules import ImpUNet
from dataset import ISICDataset
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

BATCH_SIZE = 10
LEARNING_RATE = 5*10**(-4)
NUM_EPOCH = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((512, 512))])

data = ISICDataset(img_dir="data", transform=transform)
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

total_step = len(data_loader)

model = ImpUNet(3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss()

for epoch in range(NUM_EPOCH):
    for i, img in enumerate(data_loader):
        outputs = model(img)