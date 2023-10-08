# train.py
from dataset import load_data
import modules
import torch
import torch.nn as nn
from pytorch_metric_learning import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")
# hyperparameters
num_epoch = 20
learning_rate = 0.001

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

train_loader, validation_loader, test_loader = load_data(train_path, test_path)
model = modules.Siamese()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_steps=len(train_loader)

for epoch in range(num_epoch):
    correct = 0
    train_total = 0
    model.train()
    print("training starts")
    for i, ((images1,images2), labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        output = model(images1,images2).squeeze()
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()

        pred = torch.where(output > 0.5, 1, 0)
        correct += (pred == labels).sum().item()
        train_total += labels.size(0)
        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: {}%"
                  .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item() , 100*correct/train_total))

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, ((images1,images2), labels) in enumerate(validation_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            output = model(images1,images2).squeeze()
            val_loss = criterion(output, labels.float())
            pred = torch.where(output > 0.5, 1, 0)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    print(
        f"Epoch [{epoch+1}/{num_epoch}] \
        training_loss: {loss.item():.4f}, validation_loss: {val_loss.item():.4f}, validation accuracy: {100*correct/total}%"
    )
torch.save(model,r"C:\Users\wongm\Desktop\COMP3710\project\siamesev2.pth")
