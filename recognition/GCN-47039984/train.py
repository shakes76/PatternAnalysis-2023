import torch
import matplotlib.pyplot as plt
from modules import GCN
from dataset import test_train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = test_train()
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def step_forward():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    
    x = data.x.float()
    outputs = model(x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(outputs[data.val_mask], data.y[data.val_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def train(epochs=200):
    loss_list = []
    
    # Train the model
    for epoch in range(1, epochs):
        loss = step_forward()
        loss_list.append(loss.item())
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    # Print the loss graph
    plt.plot(loss_list)
    plt.title('Training loss of 128-100-64-32 layer')
    plt.xlabel('num of epochs')
    plt.ylabel('training loss')
    plt.show()
    
    # Evaluate the model
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    print('test accuraccy:', test_acc)
    
    if test_acc > 0.946: # if accuracy is higher than previous, save model
        save_model()

def save_model():
    torch.save(model, 'GCN.pt')

train()
# save_model()