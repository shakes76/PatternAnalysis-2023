import torch
import os
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

from modules import SiameseNetwork, Classifier, TripletLoss
from dataset import ADNITrainSiameseDataset, ADNITrainClassifierDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train Siamese Network
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1155,), (0.2254,))
])
siamese_trainset = ADNITrainSiameseDataset(os.path.join(".", "AD_NC", "train"), transform)
trainloader = torch.utils.data.DataLoader(siamese_trainset, batch_size=128, shuffle=True, pin_memory=True)
print("Data load finish")

model = SiameseNetwork()
model = model.to(device)

scaler = GradScaler()

criterion = TripletLoss()
learning_rate = 0.1
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

epochs = 10

total_step = epochs * len(trainloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=learning_rate,
                                                max_momentum=0.9, total_steps=total_step)
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for anchor, positive, negative in trainloader:
        anchor, positive, negative = anchor.to(device, non_blocking=True), positive.to(device, non_blocking=True), \
            negative.to(device, non_blocking=True)
        optimiser.zero_grad()

        with autocast():
            distance_positive = model(anchor, positive)
            distance_negative = model(anchor, negative)
            loss = criterion(distance_positive, distance_negative)

        scaler.scale(loss).backward()
        scaler.step(optimiser)

        running_loss += loss.item()

        scheduler.step()
        scaler.update()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

# classifier_trainset = ADNITrainClassifierDataset("../../AD_NC/train")
