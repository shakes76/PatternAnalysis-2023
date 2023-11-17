'''
Test for classifying MNIST Dataset using VGG
'''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 5
learning_rate = 5e-3
channels = 1

#paths
path = '/home/uqscha22/'

#--------------
#Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True) #num_workers=6
total_step = len(train_loader)

testset = torchvision.datasets.MNIST(root=path+'data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False) #num_workers=6

#--------------
# Model
cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels, num_classes=10):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = self._make_layers(cfg[vgg_name], self.in_channels)
        self.classifier = nn.Linear(512, self.num_classes)

    def forward(self, x): #forward pass of the model
        out = self.features(x)
        out = out.view(out.size(0), -1) #view as 1D
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

model = VGG('VGG9', channels, num_classes=10)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#--------------
# Train the model
model.train()
print("> Training")
start = time.time() #time generation
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): #load a batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

# Test the model
print("> Testing")
start = time.time() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

print('END')