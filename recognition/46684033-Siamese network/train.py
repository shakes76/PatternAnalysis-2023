# train.py
import dataset
import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")
ADNI_class = ['0', '1']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),

])
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test = torch.utils.data.DataLoader(testset, batch_size=64)

import matplotlib

matplotlib.use('TkAgg')

plt.ion()
def plot_embeddings(embeddings, targets, plot_number, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(2):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, c=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ADNI_class)
    plt.savefig(f"plot_{plot_number+1}.png")
    plt.close()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((dataloader.batch_size * 5, 2))
        labels = np.zeros(dataloader.batch_size * 5)
        k = 0
        counter = 0
        for images, target in dataloader:

            images = images.to(device)
            embeddings[k:k + len(images)] = model.forward_once(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
            counter += 1
            if counter == 5:
                break
    return embeddings, labels


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the euclidian distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")
# hyperparameters
num_epoch = 10
learning_rate = 0.001

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

train_loader, test_loader = dataset.load_data2(train_path, test_path)
model = modules.Siamese()
model = model.to(device)

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_steps = len(train_loader)

model.train()
print("training starts")
for epoch in range(num_epoch):
    correct = 0
    train_total = 0

    for i, ((images1, images2), labels) in enumerate(train_loader):
        # BCELoss
        # optimizer.zero_grad()
        # images1 = images1.to(device)
        # images2 = images2.to(device)
        # labels = labels.to(device)
        # x = model(images1,images2).squeeze()
        # loss = criterion(x, labels.float())
        # loss.backward()
        # optimizer.step()
        #
        # pred = torch.where(x > 0.5, 1, 0)
        # correct += (pred == labels).sum().item()
        # train_total += labels.size(0)
        # if (i + 1) % 100 == 0:
        #     print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: {}%"
        #           .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item(), 100 * correct / train_total))

        # contrastive loss
        optimizer.zero_grad()
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        x, y = model(images1, images2)
        loss = criterion(x, y, labels.float())
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: %"
                  .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item()))

    train_embeddings_cl, train_labels_cl = extract_embeddings(train, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl,epoch)

    # triplet loss
    # for i, (images1, images2, images3) in enumerate(train_loader):
    #     #triplet loss
    #     optimizer.zero_grad()
    #     images1 = images1.to(device)
    #     images2 = images2.to(device)
    #     images3 = images3.to(device)
    #     x, y, z = model(images1, images2)
    #     loss = criterion(x, y, z)
    #     loss.backward()
    #     optimizer.step()
    #
    #     pred = F.pairwise_distance(x, y, keepdim=True)
    #
    #     correct += (pred == labels).sum().item()
    #     train_total += labels.size(0)
    #     if (i + 1) % 100 == 0:
    #         print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: {}%"
    #               .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item(), 100 * correct / train_total))

    # model.eval()
    #
    # val_loss = 0.0
    # correct = 0
    # total = 0
    #
    # with torch.no_grad():
    #     for i, ((images1,images2), labels) in enumerate(test_loader):
    #         images1 = images1.to(device)
    #         images2 = images2.to(device)
    #         labels = labels.to(device)
    #         output = model(images1,images2).squeeze()
    #         val_loss = criterion(output, labels.float())
    #
    #         pred = torch.where(output > 0.5, 1, 0)
    #         correct += (pred == labels).sum().item()
    #         total += labels.size(0)
    #
    # print(
    #     f"Epoch [{epoch + 1}/{num_epoch}] \
    #             training_loss: {loss.item():.4f}, validation_loss: {val_loss.item():.4f}, validation accuracy: {100 * correct / total}%"
    # )
