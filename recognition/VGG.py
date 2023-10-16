"""
    File name: modules.py
    Author: Fanhao Zeng
    Date created: 11/10/2023
    Date last modified: 12/10/2023
    Python Version: 3.10.12
"""


import torch
from torch import nn
from torch.hub import load_state_dict_from_url


def make_layers(cfg, batch_norm=False, in_channels=3):
    """
    Make layers for VGG
    :param config: configuration of VGG
    :param batch_norm: whether to use batch normalization
    :param in_channels: number of input channels
    :return: layers

    E.g. 256,256,3 -> 256,256,64 -> 128,128,64 -> 128,128,128 -> 64,64,128 -> 64,64,256 -> 32,32,256 -> 32,32,256 ->
    16,16,256 -> 16,16,512 -> 8,8,512 -> 8,8,512 -> 4,4,512

    4 * 4 * 512 = 8192
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*layers)


cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        """
        VGG constructor
        :param features: features of VGG
        :param num_classes: number of classes

        """
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass for VGG
        :param x: input
        :return: output
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):

        """
        Initialize weights for VGG
        :return: None
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def VGG16(pretrained, in_channels, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    return model