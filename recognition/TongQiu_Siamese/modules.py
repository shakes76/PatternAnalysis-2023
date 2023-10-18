import torch
import torch.nn as nn

"""
Base Line Model
"""


class Baseline_Contrastive(nn.Module):
    def __init__(self):
        super(Baseline_Contrastive, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@231x247
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@115x123
            nn.Conv2d(64, 128, 7),  # 128@109x117
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128@54x58
            nn.Conv2d(128, 128, 4),  # 128@51x55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128@25x27
            nn.Conv2d(128, 256, 4),  # 256@22x24
            nn.ReLU(inplace=True),
        )
        self.liner = nn.Sequential(nn.Linear(135168, 4096), nn.Sigmoid())  # nn.Linear(9216, 4096)

    def sub_forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        x1 = self.sub_forward(x1)
        x2 = self.sub_forward(x2)
        return x1, x2


class Baseline_Triplet(nn.Module):
    pass


# test if the model works properly
if __name__ == '__main__':
    model = Baseline_Contrastive()
    test_tensor = torch.ones(3, 1, 240, 256)
    output = model(test_tensor, test_tensor)
    print(output[0].shape)
    print(output[1].shape)
