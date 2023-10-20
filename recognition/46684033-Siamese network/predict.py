#predict.py
import torch
import dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import modules
import dataset
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchsummary import summary
if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomCrop(128, 16),
    transforms.RandomRotation(degrees=(-20, 20)),
    #transforms.RandomPerspective(distortion_scale=0.6,p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])
batch_size = 64
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
validation_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/validation"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

trainset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
validationset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
testset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = True)



model = torch.load(f"C:/Users/wongm/Desktop/COMP3710/project/siamese_1.5_epoch_49.pth")
model = model.to(device)
model.eval()

learning_rate = 0.00005
classifier = modules.Classifier()
classifier = classifier.to(device)
classifier_loss = nn.CrossEntropyLoss()
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-3)

correct = 0
total = 0
num_epoch = 30
classifier.train()
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
        for i, (test_image, test_label) in enumerate(test_loader):
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
    torch.save(model, f"C:\\Users\\wongm\\Desktop\\COMP3710\\project\\classifier_{epoch + 1}.pth")


epochs = list(range(1, num_epoch + 1))
# Plot both training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_accuracy, linestyle='-', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, linestyle='-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"Accuracy_plot.png")
plt.show(block=True)

plt.figure(figsize=(8, 6))
plt.plot(epochs, training_loss, linestyle='-', label='Training loss')
plt.plot(epochs, validation_loss, linestyle='-', label='Validation loss')
plt.title('Training and Validation loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"Loss_plot.png")
plt.show(block=True)



















    #TODO
    # test all model in certain epoch
    # include average result