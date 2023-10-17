
import torch
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, train_loader, optimizer, criterion):
    model.train()
    for batch in train_loader:
        data, target = batch['data'], batch['label']
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            data, target = batch['data'], batch['label']
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(val_loader.dataset), 100. * accuracy))

def test_model(model, test_loader):
    model.eval()
    # Testing loop logic

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print("Model saved successfully.")