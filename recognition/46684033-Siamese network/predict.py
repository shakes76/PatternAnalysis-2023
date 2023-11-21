#predict.py
import torch
import torchvision
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not torch.cuda.is_available():
    print("Warning VUDA not Found. Using CPU")


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

classifier = torch.load(f"C:/Users/wongm/Desktop/COMP3710/project/classifier_1.pth")
classifier = classifier.to(device)
classifier.eval()

print("Testing start")
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

t_correct = 0
t_total = 0
tc_loss = 0
test_acc = 0

with torch.no_grad():
    for i, (test_image, test_label) in enumerate(test_loader):
        test_image = test_image.to(device)
        test_label = test_label.to(device)
        embeddings = model.forward_once(test_image)
        output = classifier(embeddings).squeeze()

        _, pred = torch.max(output.data, 1)
        t_correct += (pred == test_label).sum().item()
        t_total += test_label.size(0)

        test_acc = (100 * t_correct / t_total)
        if (i%20) == 0 :
            print(f"Steps[{i}/{len(test_loader)}] accumulated test accuracy: {test_acc}%")
print(f"Test Accuracy is {test_acc}%")