import torch.nn as nn
import torch.nn.functional as F
import dataset as ds
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Resnet Class (50 maybe or 25)
class BasicBlock(nn.Module):
    expansion =1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self,  block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 100
        
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(100)
        self.layer1 = self._make_layer(block, 100, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 250, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 300, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 750, num_blocks[3], stride=2)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def ResNet34():
        return ResNet(BasicBlock, [2, 2, 2, 2])


network = ResNet.ResNet34()
network.to(device=device)

dataset = ds.ADNI_Dataset()
train_loader = dataset.get_train_loader()

network.train()
for j, (images, labels) in  enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = network(images)
    print(outputs.shape)

# Perceiver class (or import)

#make ADNI class that merges the previous two