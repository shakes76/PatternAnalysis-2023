import torch
import torch.nn as nn
import modules
import dataset
import time

NUM_EPOCHS = 100
# Placeholder for now
PATH = "model.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# get model (GCN)
model = modules.GCN(in_channels=128, num_classes=4)
model = model.to(device)

# get dataset
dataset = dataset.data

print('starting test')

# need to train and then save to be able to use her for the predict file
# Set optimizer and riterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
model.train()
start = time.time()
for epoch in range(NUM_EPOCHS):
    #Might want to change to batches instead
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    loss = criterion(out, dataset.y)
    loss.backward()
    optimizer.step()

    if (epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f}\n', flush=True)
        

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Test the model
model.eval()
start = time.time()
out = model(dataset.x, dataset.edge_index)
pred = out.argmax(dim=1)
test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]
test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum()) 
print("Test Accuracy: " + str(test_acc))
    
# Check if this works
torch.save(model,PATH)

