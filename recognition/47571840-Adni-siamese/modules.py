import torch.nn as nn
import torch.nn.functional as F
import torch

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_shape):
        super(EmbeddingNetwork, self).__init__()

        #number of channels, number filters, kernel size
        self.conv1 = nn.Conv2d(input_shape[0], 32, 10)
        self.conv2 = nn.Conv2d(32, 64, 7)
        self.conv3 = nn.Conv2d(64, 64, 4)
        self.conv4 = nn.Conv2d(64, 128, 4)
        self.fc1 = nn.Linear(128*22*24, 50)# This assumes a specific flattened size from previous layers
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.bn(x)
        x = F.relu(self.conv4(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, input_shape):
        super(SiameseNetwork, self).__init__()

        self.embedding_net = EmbeddingNetwork(input_shape)
        self.fc = nn.Linear(1, 1)

    def forward_one(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        distance = F.pairwise_distance(output1, output2, p=2)
        out = torch.sigmoid(self.fc(distance.unsqueeze(1)))
        return out


'''
Alright, let's walk through the shape transformation of the tensor as it passes through the `EmbeddingNetwork`.

1. **Initial Input:** Shape = `[batch_size, channels, 240, 256]`. Assuming grayscale images, channels will be 1.

2. **After `conv1`:** Kernel size = 10 and stride = 1, so:
   Output shape = `[batch_size, 32, 231, 247]` (Because 240-10+1=231 and 256-10+1=247).

3. **After first max pooling:** Pool size = 2:
   Output shape = `[batch_size, 32, 115, 123]`.

4. **After `conv2`:** Kernel size = 7:
   Output shape = `[batch_size, 64, 109, 117]` (Because 115-7+1=109 and 123-7+1=117).

5. **After second max pooling:** Pool size = 2:
   Output shape = `[batch_size, 64, 54, 58]`.

6. **After `conv3`:** Kernel size = 4:
   Output shape = `[batch_size, 64, 51, 55]`.

7. **After third max pooling:** Pool size = 2:
   Output shape = `[batch_size, 64, 25, 27]`.

8. **After `conv4`:** Kernel size = 4:
   Output shape = `[batch_size, 128, 22, 24]`.

'''