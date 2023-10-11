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
test = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True)

import matplotlib

matplotlib.use('TkAgg')

plt.ion()
def plot_embeddings(embeddings, targets, plot_number,plot_name, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(2):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, c=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ADNI_class)
    plt.savefig(f"{plot_name}_{plot_number+1}.png")
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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


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
num_epoch = 60
learning_rate = 0.001

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

train_loader, validation_loader, test_loader = dataset.load_data2(train_path, test_path)
model = modules.Siamese()
model = model.to(device)

criterion = ContrastiveLoss(2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_steps = len(train_loader)

model.train()
print("training starts")

training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []
loss = 0
for epoch in range(num_epoch):
    correct = 0
    train_total = 0
    total_loss_this_epoch = []
    total_val_loss_this_epoch = []
    model.train()
    #for BCELoss and contrastive loss
    for i, ((images1, images2), labels) in enumerate(train_loader):
    #     # BCELoss
    #     # optimizer.zero_grad()
    #     # images1 = images1.to(device)
    #     # images2 = images2.to(device)
    #     # labels = labels.to(device)
    #     # x = model(images1,images2).squeeze()
    #     # loss = criterion(x, labels.float())
    #     # loss.backward()
    #     # optimizer.step()
    #     #
    #     # pred = torch.where(x > 0.5, 1, 0)
    #     # correct += (pred == labels).sum().item()
    #     # train_total += labels.size(0)
    #     # if (i + 1) % 100 == 0:
    #     #     print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: {}%"
    #     #           .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item(), 100 * correct / train_total))
    #
    #     # contrastive loss
        optimizer.zero_grad()
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        x, y = model(images1, images2)
        loss = criterion(x, y, labels.float())
        loss.backward()
        optimizer.step()

        #accuracy test
        distances = (y - x).pow(2).sum(1)
        #average loss in this epoch
        total_loss_this_epoch.append(loss.item())
        if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: %"
                      .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item()))


    training_loss.append(sum(total_loss_this_epoch)/len(total_loss_this_epoch))


    train_embeddings_cl, train_labels_cl = extract_embeddings(train, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl,epoch,"trainingv2")

    test_embeddings_cl, test_labels_cl = extract_embeddings(test, model)
    plot_embeddings(test_embeddings_cl, test_labels_cl, epoch, "testv2")

    # triplet loss
    # for i, (images1, images2, images3) in enumerate(train_loader):
    #
    #     optimizer.zero_grad()
    #     images1 = images1.to(device)
    #     images2 = images2.to(device)
    #     images3 = images3.to(device)
    #     x, y, z = model(images1, images2, images3)
    #     loss = criterion(x, y, z)
    #     loss.backward()
    #     optimizer.step()
    #
    #
    #
    #     if (i + 1) % 100 == 0:
    #         print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} Accuracy: {}%"
    #               .format(epoch + 1, num_epoch, i + 1, total_steps, loss.item(), 100 * correct / train_total))
    # train_embeddings_cl, train_labels_cl = extract_embeddings(train, model)
    # plot_embeddings(train_embeddings_cl, train_labels_cl, epoch)

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, ((images1,images2), labels) in enumerate(validation_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            x,y = model(images1,images2)
            val_loss = criterion(x,y, labels.float())
            total_val_loss_this_epoch.append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epoch}], training_loss: {loss.item():.4f}, validation_loss: {val_loss.item():.4f}"
    )

    validation_loss.append(sum(total_val_loss_this_epoch)/len(total_val_loss_this_epoch))
    if min(validation_loss) > val_loss.item():
        torch.save(model,r"C:\Users\wongm\Desktop\COMP3710\project\siamese.pth")
        print("new min val_loss, model saved")

epochs = list(range(1, num_epoch + 1))
# Plot both training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_loss,linestyle='-', label='Training Loss')
plt.plot(epochs, validation_loss, linestyle='-', label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"loss_plot .png")
plt.show(block=True)

#TODO
#test metric = 1.0
#test different learning rate
#test accuracy after saving good model
