from dataset import get_dataloader
import torch
import time
from torch import nn

from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# Hyper parameters
BATCH_SIZE = 64
EPOCHS = 3
LR = 1e-5

# Initialise data loaders
train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)

# Initialise model, criterion and optimiser
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.heads.head = nn.Linear(in_features=BATCH_SIZE*12, out_features=2)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training and validation
start = time.time()
for epoch in range(EPOCHS):
    # Do training
    train_correct = 0
    train_total = 0
    train_losses = []
    strt = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        #losses
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i+1) % 27  == 0:
            print (f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}] Loss: {loss.item()}, {train_correct/train_total} accuracy")
            
    print (f"Training epoch [{epoch+1}/{EPOCHS}], {train_correct/train_total} accuracy. Time elapsed {time.time()-strt}.")
        
    # Do validation
    val_correct = 0
    val_total = 0
    val_losses = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            #losses
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            if (i+1) % 6 == 0:
                print (f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(val_loader)}] Loss: {loss.item()}, {val_correct/val_total} accuracy")
            
    print (f"Validation epoch [{epoch+1}/{EPOCHS}], {val_correct/val_total} accuracy")

end = time.time()
elapsed = end - start
print("Training & validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Do testing
test_correct = 0
test_total = 0
start = time.time()
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f"Test accuracy {test_correct/test_total}, time elapsed {time.time()-start}")
