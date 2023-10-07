# train.py
from dataset import load_data
import modules
import torch
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
model = modules.ResNet18()
model = model.to(device)

loss_func = losses.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_steps=len(train_loader)
for epoch in range(num_epoch):
    model.train()
    print("training starts")
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        embeddings = model(images)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f}"
                  .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item()))

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            val_loss = loss_func(embeddings, labels)

    print(
        f"Epoch [{epoch}/{num_epoch}] \
        training_loss: {loss.item():.4f}, validation_loss: {val_loss.item():.4f}"
    )
torch.save(model,r"C:/Users/wongm/Desktop/COMP3710/project/")
