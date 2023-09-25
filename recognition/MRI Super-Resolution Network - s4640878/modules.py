import sys, os, time
import torch, torchvision
from traceback import format_exc


class BasicBlock(torch.nn.Module):
    """
    basic building block for the ResNet image classifier
    
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        init BasicBlocks class 
        
        """
        super(BasicBlock, self).__init__()

        """ first 2d conv layer """
        self._conv_1 = torch.nn.Conv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=3, 
            stride=stride, padding=1, bias=False,
        )
        self._bn_1 = torch.nn.BatchNorm2d(num_features=planes)

        """ second 2d conv layer """
        self._conv_2 = torch.nn.Conv2d(
            in_channels=planes, out_channels=planes, kernel_size=3, 
            stride=1, padding=1, bias=False,
        )
        self._bn_2 = torch.nn.BatchNorm2d(num_features=planes)

        """ shortcut used for back prop to reduce diminishing gradient """
        self._shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self._shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_planes, out_channels=self.expansion*planes,
                    kernel_size=1, stride=stride, bias=False,
                ),
                torch.nn.BatchNorm2d(num_features=self.expansion*planes),
            )

    def forward(self, x):
        """
        forward pass

        """
        out = torch.nn.functional.relu(self._bn_1(self._conv_1(x)))
        out = self._bn_2(self._conv_2(out))
        out += self._shortcut(x)

        return torch.nn.functional.relu(out)
    

class ResNet(torch.nn.Module):
    """
    generic ResNet convolutional neural network

    """

    def __init__(self, block, num_blocks, num_classes=2):
        """
        init ResNet class

        """
        super(ResNet, self).__init__()
        self._in_planes = 64

        """ initial 2d conv layer """
        self._conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self._bn_1 = torch.nn.BatchNorm2d(num_features=64)

        """ string together BasicBlocks """
        self._layer_1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self._layer_2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self._layer_3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self._layer_4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        """ final linear / fully connected layer """
        self._linear = torch.nn.Linear(
            in_features=28_672*block.expansion, out_features=num_classes,
        )

    def make_layer(self, block, planes, num_blocks, stride):
        """
        creates a single layer based in the BasicBlocks params provided

        """
        strides, layers = [stride] + [1]*(num_blocks - 1), []
        
        for stride in strides:
            layers.append(block(self._in_planes, planes, stride))
            self._in_planes = planes*block.expansion

        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """
        forward pass

        """
        out = torch.nn.functional.relu(self._bn_1(self._conv_1(x)))
        out = self._layer_4(self._layer_3(self._layer_2(self._layer_1(out))))
        out = torch.nn.functional.avg_pool2d(out, 4)

        return torch.nn.functional.softmax(self._linear(out.view(out.size(0), -1)), -1)
    

def ResNet18():
    """
    18 layer ResNet conv neural network

    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    """
    34 layer ResNet conv neural network

    """
    return ResNet(BasicBlock, [3, 4, 6, 3])