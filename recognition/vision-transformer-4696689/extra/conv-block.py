"""
Conv v2
"""
class ConvLayer2(nn.Module):
    def __init__(self):
        super().__init__()
        #pool
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        #first layer
        self.conv11_x = nn.Conv2d(20, 48, kernel_size=(11,11), stride=(4,4), padding=(0,0))
        self.conv11_y = nn.Conv2d(240, 48, kernel_size=(11,3), stride=(4,1), padding=(0,0))
        self.conv11_z = nn.Conv2d(256, 48, kernel_size=(3,11), stride=(1,4), padding=(0,0))
        #second layer
        self.conv5_x = nn.Conv2d(48, 192, kernel_size=(5,5), stride=(2,2), padding=(0,0))
        self.conv5_y = nn.Conv2d(48, 192, kernel_size=(5,3), stride=(2,1), padding=(0,0))
        self.conv5_z = nn.Conv2d(48, 192, kernel_size=(3,5), stride=(1,2), padding=(0,0))
        #projection
        self.l_x = nn.Linear(30, 32)
        self.l_y = nn.Linear(12, 32)
        self.l_z = nn.Linear(10, 32)

    def forward(self, imgs):
        #input N, C, L, W, H
        #first layer
        x_x = self.relu(self.pool(self.conv11_x(imgs.flatten(1,2))))
        x_y = self.relu(self.pool(self.conv11_y(imgs.permute(0,1,3,4,2).flatten(1,2))))
        x_z = self.relu(self.pool(self.conv11_z(imgs.permute(0,1,4,2,3).flatten(1,2))))
        #second layer
        x_x = self.relu(self.pool(self.conv5_x(x_x)))
        x_y = self.relu(self.pool(self.conv5_y(x_y)))
        x_z = self.relu(self.pool(self.conv5_z(x_z)))
        #projection
        x_x = self.l_x(x_x.flatten(2,3))
        x_y = self.l_y(x_y.flatten(2,3))
        x_z = self.l_z(x_z.flatten(2,3))
        return torch.cat([x_x, x_y, x_z], dim=2)