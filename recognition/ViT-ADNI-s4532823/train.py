from dataset import get_dataloader
import torch
import time

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
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training and validation
start = time.time()
for epoch in range(EPOCHS):
    # Do training
    for i, (images, label) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        something = model(images)

        #losses
        loss = None
        accuracy = None

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100  == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, EPOCHS, i+1, len(train_loader), loss.item()))
            
    # Do validation
    for i, (images, label) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

end = time.time()
elapsed = end - start
print("Training & validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Do testing
with torch.no_grad():
    for i, (images, label) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)