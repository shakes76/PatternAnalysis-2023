# train.py
from dataset import load_data
import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")
# hyperparameters
num_epoch = 4
learning_rate = 0.001

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

train_loader, validation_loader, test_loader = load_data(train_path, test_path)
model = modules.Siamese()
model = model.to(device)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_steps=len(train_loader)

print("training starts")
for epoch in range(num_epoch):
    correct = 0
    train_total = 0
    model.train()

    for i, ((images1,images2), labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        output1,output2 = model(images1,images2)
        loss = criterion(output1,output2, labels.float())
        loss.backward()
        optimizer.step()


        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} "
                  .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item() ))

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, ((images1,images2), labels) in enumerate(validation_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            output1,output2 = model(images1,images2)
            val_loss = criterion(output1,output2, labels.float())


    print(
        f"Epoch [{epoch+1}/{num_epoch}] \
        training_loss: {loss.item():.4f}, validation_loss: {val_loss.item():.4f}"
    )
torch.save(model,r"C:\Users\wongm\Desktop\COMP3710\project\siamesev2.pth")
