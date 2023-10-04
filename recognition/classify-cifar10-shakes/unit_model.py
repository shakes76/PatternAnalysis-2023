'''
Unit test for displaying a basic model
'''
import torch.nn as nn

# Hyper-parameters
channels = 1

#--------------
#Data
print("> Setup dataset")
# Insert data here if building actual training of model

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

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)
