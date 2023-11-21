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

#For visualization
ADNI_class = ['0', '1']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']

batch_size = 16


import matplotlib

matplotlib.use('TkAgg')

plt.ion()
def plot_embeddings(embeddings, targets, plot_number, plot_name, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(2):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, c=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ADNI_class)
    plt.savefig(f"./graphs/{plot_name}_{plot_number + 1}.png")
    plt.close()


def extract_embeddings(dataloader, model):
    no_of_batch = 40
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((dataloader.batch_size * no_of_batch, 128))
        labels = np.zeros(dataloader.batch_size * no_of_batch)
        k = 0
        counter = 0
        for (images, images2, images3), labela, labelb, target in dataloader:

            images = images.to(device)
            embeddings[k:k + len(images)] = model.forward_once(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
            counter += 1
            if counter == no_of_batch:
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
        self.eps = 1e-9 # add a small constant to prevent numerical instability when square root

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")
# hyperparameters
num_epoch = 90
learning_rate = 0.0001

train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
validation_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/validation"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"


train_loader, validation_loader, test_loader = dataset.load_data2(train_path, validation_path, test_path)
model = modules.Siamese()
model = model.to(device)

#loss and optimizer choice
criterion = ContrastiveLoss(1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

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
    x_embeddings = []
    y_embeddings = []
    # #for contrastive loss
    for i, ((images1, images2, images3), labela, labelb, test_label) in enumerate(train_loader):

        # First train the model with a pair of training image and positive image
        optimizer.zero_grad()
        images1 = images1.to(device)
        images2 = images2.to(device)
        labela = labela.to(device)
        x, y = model(images1, images2)
        loss = criterion(x, y, labela.float())
        loss.backward()
        optimizer.step()
        total_loss_this_epoch.append(loss.item())

        # then train the model with a pair of training image and negative iamge
        optimizer.zero_grad()
        images1 = images1.to(device)
        images3 = images3.to(device)
        labelb = labelb.to(device)
        x, y = model(images1, images3)
        loss = criterion(x, y, labelb.float())
        loss.backward()
        optimizer.step()


        # average loss in this epoch
        total_loss_this_epoch.append(loss.item())
        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step[{}/{}] Loss: {:.5f} "
                  .format(epoch + 1, num_epoch, i + 1, total_steps,
                          sum(total_loss_this_epoch) / len(total_loss_this_epoch)))

    training_loss.append(sum(total_loss_this_epoch) / len(total_loss_this_epoch))

    # contrastive loss validation
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, ((images1, images2, images3), labela, labelb, test_label) in enumerate(validation_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)
            labela = labela.to(device)
            x, y = model(images1, images2)
            val_loss = criterion(x, y, labela.float())
            total_val_loss_this_epoch.append(val_loss.item())

            images1 = images1.to(device)
            images3 = images3.to(device)
            labelb = labelb.to(device)
            x, y = model(images1, images3)
            val_loss = criterion(x, y, labelb.float())
            total_val_loss_this_epoch.append(val_loss.item())

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step[{}/{}] Validation Loss: {:.5f} "
                      .format(epoch + 1, num_epoch, i + 1, len(validation_loader),
                              sum(total_val_loss_this_epoch) / len(total_val_loss_this_epoch)))


    print(
        f"Epoch [{epoch + 1}/{num_epoch}], training_loss: {sum(total_loss_this_epoch) / len(total_loss_this_epoch):.4f}, validation_loss: {sum(total_val_loss_this_epoch) / len(total_val_loss_this_epoch):.4f}"
    )

    validation_loss.append(sum(total_val_loss_this_epoch) / len(total_val_loss_this_epoch))

    torch.save(model, f"C:\\Users\\wongm\\Desktop\\COMP3710\\project\\siamese_epoch_{epoch + 1}.pth")

    #plot embeddings
    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl, epoch, "training")

    train_embeddings_cl, train_labels_cl = extract_embeddings(validation_loader, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl, epoch, "validation")

    test_embeddings_cl, test_labels_cl = extract_embeddings(test_loader, model)
    plot_embeddings(test_embeddings_cl, test_labels_cl, epoch, "test")

    epochs = list(range(0, epoch + 1))
    # Plot both training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, linestyle='-', label='Training Loss')
    plt.plot(epochs, validation_loss, linestyle='-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss_plot_epoch_{epoch}.png")
    plt.close()

#Train a classifier
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import modules
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchsummary import summary
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")

# data augmentation
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomCrop(128, 16),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
#Hyperparameters
batch_size = 64
learning_rate = 0.00005
#Paths
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
validation_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/validation"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

#create data loaders
trainset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
validationset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
testset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = True)



#load Siamese neural network to get the embeddings
model = torch.load(f"C:/Users/wongm/Desktop/COMP3710/project/siamese_1.5_epoch_49.pth")
model = model.to(device)
model.eval()

#create a classifier instance
classifier = modules.Classifier()
classifier = classifier.to(device)

#loss and optimizer selection
classifier_loss = nn.CrossEntropyLoss()
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-3)

correct = 0
total = 0
num_epoch = 30
classifier.train()

#print out the classifier structure
summary(classifier, input_size=(1,128), batch_size=64)
print("classifier training start")
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []
for epoch in range(num_epoch):
    c_correct = 0
    c_total = 0
    t_correct = 0
    t_total = 0
    classifier.train()
    c_loss = 0
    tc_loss = 0
    train_acc = 0
    test_acc = 0
    total_loss_this_epoch = []
    total_val_loss_this_epoch = []
    for i,(image,label) in enumerate(train_loader):
        image=image.to(device)
        label=label.to(device)
        # train classifier
        classifier_optimizer.zero_grad()
        test_label = label.to(device)
        embeddings = model.forward_once(image)
        output = classifier(embeddings).squeeze()
        c_loss = classifier_loss(output, label)
        c_loss.backward()
        classifier_optimizer.step()

        _, pred = torch.max(output.data, 1)
        c_correct += (pred == label).sum().item()
        c_total += label.size(0)
        train_acc = 100 * c_correct / c_total
        total_loss_this_epoch.append(c_loss.item())

        if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step[{}/{}] Training accuracy: {}% "
                      .format(epoch + 1, num_epoch, i + 1, len(train_loader), train_acc))
    training_accuracy.append(train_acc)
    training_loss.append(sum(total_loss_this_epoch) / len(total_loss_this_epoch))

    classifier.eval()
    with torch.no_grad():
        for i, (test_image, test_label) in enumerate(validation_loader):
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            embeddings = model.forward_once(test_image)
            output = classifier(embeddings).squeeze()
            tc_loss = classifier_loss(output,test_label)

            _, pred = torch.max(output.data, 1)
            t_correct += (pred == test_label).sum().item()
            t_total += test_label.size(0)
            test_acc = (100 * t_correct / t_total)
            total_val_loss_this_epoch.append(tc_loss.item())
            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step[{}/{}] test Accuracy: {}% "
                      .format(epoch + 1, num_epoch, i + 1, len(validation_loader), test_acc))
    validation_accuracy.append(test_acc)
    validation_loss.append(sum(total_val_loss_this_epoch) / len(total_val_loss_this_epoch))
    print("Epoch [{}/{}], Training Loss: {:.5f} Training Accuracy: {:.5f}% Testing Loss: {:.5f} Validation accuracy: {:.5f}%"
          .format(epoch + 1, num_epoch, c_loss.item(), train_acc, tc_loss.item(),test_acc))
    torch.save(classifier, f"C:\\Users\\wongm\\Desktop\\COMP3710\\project\\classifier_{epoch + 1}.pth")

epochs = list(range(1, num_epoch + 1))

# Plot both training and validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_accuracy, linestyle='-', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, linestyle='-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"Accuracy_plot.png")
plt.show(block=True)

# Plot both training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_loss, linestyle='-', label='Training loss')
plt.plot(epochs, validation_loss, linestyle='-', label='Validation loss')
plt.title('Training and Validation loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"Loss_plot.png")
plt.show(block=True)