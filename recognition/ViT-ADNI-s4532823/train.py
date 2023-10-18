"""
Train the model and perform testing.
"""

from dataset import get_dataloader
import torch
import time
from torch import nn
from modules import ViT
import math
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# Hyper parameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-5

# Initialise data loaders
train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)

# Initialise model, criterion and optimiser
model = ViT()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.5)

# Training and validation
train_losses = []
val_losses = []
train_accs = []
val_accs = []
stopping_epoch = EPOCHS

start = time.time()
for epoch in range(EPOCHS):
    # Do training
    train_correct = 0
    train_total = 0
    tl = []
    strt = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        #losses
        loss = criterion(outputs, labels)
        tl.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

    train_losses.append(sum(tl)/len(tl)) # mean training loss over current epoch
    train_accs.append(train_correct/train_total) # training accuracy over current epoch
    print (f"Training epoch [{epoch+1}/{EPOCHS}]: mean loss {train_losses[-1]}, accuracy {train_accs[-1]}. Time elapsed {time.time()-strt}.")
    print("")

    # Do validation
    val_correct = 0
    val_total = 0
    vl = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            #losses
            loss = criterion(outputs, labels)
            vl.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_losses.append(sum(vl)/len(vl)) # mean validation loss over current epoch
    val_accs.append(val_correct/val_total) # validation accuracy over 

    print (f"Validation epoch [{epoch+1}/{EPOCHS}]: mean loss {val_losses[-1]}, accuracy {val_accs[-1]}.")
    print("")
    print("")

    scheduler.step()

    # Stop training early if validation accuracy decreases for two epochs in a row
    if epoch+1 > 1 and val_accs[-3] > val_accs[-2] > val_accs[-1]:
        stopping_epoch = epoch+1
        break
    if val_accs[-1] == max(val_accs):
        # save best model
        torch.save(model, "adni_vit.pt")
    
end = time.time()
elapsed = end - start
print("Training & validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
print("")
print("")


plt.plot(range(1, stopping_epoch+1), train_accs, label='Training accuracy')
plt.plot(range(1, stopping_epoch+1), val_accs, label='Validation accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training accuracy vs validation accuracy")

# Show the figure with both subplots
plt.show()
plt.savefig(f"Training vs validation accuracy {time.strftime(time.time())}")

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
