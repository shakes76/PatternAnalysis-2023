import os.path

import torch
from modules import *
from dataset import *
from train import *

# Build Pixel CNN
class MaskedConv2d(nn.Conv2d):

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.mask_type = mask_type

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        mask = torch.zeros(kernel_size)
        mask[:kernel_size[0] // 2, :] = 1.0
        mask[kernel_size[0] // 2, :kernel_size[1] // 2] = 1.0
        if self.mask_type == "B":
            mask[kernel_size[0] // 2, kernel_size[1] // 2] = 1.0
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        self.weight.data *= self.mask  # mask weights
        return super().forward(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1),
            nn.ReLU(inplace=True)
        )
        # masked conv2d
        self.conv2 = nn.Sequential(
            MaskedConv2d("B", in_channel // 2, in_channel // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel // 2, in_channel, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return inputs + x


class PixelCNN(nn.Module):

    def __init__(self, in_channel=1, channels=128, out_channel=1, num_residual_block=5):
        super(PixelCNN, self).__init__()
        self.down_size = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.stem = nn.Sequential(
            MaskedConv2d("A", in_channel, channels, 7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_block)]
        )
        self.head = nn.Sequential(
            MaskedConv2d("B", channels, channels, 2, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", channels, channels, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_channel, 3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.head(x)
        x = self.down_size(x)
        return x


# Training Pixel CNN

def train_pixel():
    if not os.path.exists("./Model"):
        os.mkdir("./Model")
    model_ = PixelCNN().to(device)
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM,
                  COMMITMENT_COST).to(device)
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=1e-3)

    tqdm_bar = tqdm(range(10000))
    pre_loss = 1.0
    epochs = 0
    for eq in tqdm_bar:

        train_img_, _ = next(iter(training_loader))
        train_img = train_img_.to(device)
        _, _, _, quantized = model(train_img)
        output = model_(train_img)
        print('quantized',quantized.shape)
        print('output',output.shape)
        exit()
        loss = F.mse_loss(output, quantized)

        loss.backward()
        optimizer_.step()
        optimizer_.zero_grad()
        epochs += 1
        if epochs % 10 == 0:
            tqdm_bar.set_description('loss: {}'.format(loss))

        with torch.no_grad():
            if loss <= pre_loss:
                print("Saving model...")
                torch.save(model_.state_dict(), "./Model/pixelCNN.pth")
                pre_loss = loss

if __name__ == "__main__":
    train_pixel()