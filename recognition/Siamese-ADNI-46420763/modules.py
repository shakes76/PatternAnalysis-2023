import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basic Convlutional Block
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
class SiameseNetwork(nn.Module):
    """
    Siamese Network Model
    """
    def __init__(self, layers = [1, 64, 128, 128, 256], kernel_sizes = [10, 7, 4, 4]):
        super(SiameseNetwork, self).__init__()
        self.layers = layers
        self.blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1], kernel_size=kernel_sizes[i]) for i in range(len(layers) - 1)])
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 7, 64),  #256 * 8 * 7 # 123648
            #nn.Dropout(inplace=True)
        )
        self.dense = nn.Linear(64, 1)
        
    def forward_once(self, x):
        """
        Creates embedding vector of input image
        """
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i != len(self.blocks) - 1:
                x = self.maxpool(x)
        x = self.linear(x)
        x = F.sigmoid(x)
        return x
    
    def forward(self, x1, x2):
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        distance = torch.abs(embedding1 - embedding2)
        
        output = self.dense(distance)
        output = F.sigmoid(output)
        return output
    

# Model
class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride = 1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_planes, planes, kernel_size = 3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    # Convolve X to match output the channel number and image size of conv2
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion*planes,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x) # add X to the output of the conv layers
    out = F.relu(out)
    return out

# ResNet
class ResNet(nn.Module):
  def __init__(self, block, num_blocks, embedding_size=256):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(28672, embedding_size) # 6144

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
      out = self.linear(out)
      return out
  
def ResNetEmbedder():
  return ResNet(BasicBlock, [2, 2, 2, 2])
