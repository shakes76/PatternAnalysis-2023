'''
Train the model from modules.py with the data from dataset.py and save the model
'''
import torch
import torch.nn as nn
import modules
import dataset
import time
import matplotlib.pyplot as plt
import numpy as np

NUM_EPOCHS = 100
# Save the model to the file model.pt
PATH = "model.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# get model (GCN)
model = modules.GCN(in_channels=128, num_classes=4)
model = model.to(device)

# get dataset
data = dataset.dataset
print('starting test')

# Set optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
model.train()
start = time.time()
# Set up empty lists to store the epoch and the train loss
epoch_val = []
loss_val = []

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Add datapoint to the graph
    epoch_val.append(epoch)
    loss_val.append(loss.item())

    if (epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f}\n', flush=True)
        
# plot the train loss against the epoch
plt.title("Training Loss Function")
plt.plot(epoch_val,loss_val)
plt.xlabel("Epoch")
plt.ylabel("Loss")

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Test the model
model.eval()
start = time.time()

out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)
# Find the number of nodes that were correctly classified and then calculate the percentage for accuracy
test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
test_acc = int(test_correct) / int(data.test_mask.sum().item()) 
print("Test Accuracy: " + str(test_acc))
    
# Save the model to model.pt
torch.save(model,PATH)

# Show the loss function plot after all the calculations have finished
plt.show()


