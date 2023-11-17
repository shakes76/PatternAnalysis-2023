import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net model for image segmentation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        depth (int): Depth of the U-Net architecture.
        base_filters (int): Number of filters in the first layer. It doubles with each layer.
        dropout_prob (float): Dropout probability.
    """
    def __init__(self, in_channels, out_channels, depth=4, base_filters=64, dropout_prob=0.3):
        super(UNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_prob)
        self.upsamples = nn.ModuleList()

        # Contracting path
        for i in range(depth):
            in_ch = in_channels if i == 0 else base_filters * 2**i
            out_ch = base_filters * 2**(i+1)
            self.encoders.append(self.conv_block(in_ch, out_ch))

        # Expanding path
        for i in range(depth-1, 0, -1):
            in_ch = base_filters * 2**(i+1)  # After concatenation
            out_ch = base_filters * 2**i
            self.decoders.append(self.conv_block(in_ch, out_ch))
            self.upsamples.append(nn.ConvTranspose2d(out_ch, out_ch//2, kernel_size=2, stride=2))

        self.decoders.append(self.conv_block(base_filters * 2, base_filters))
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor of the network.
        """
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        rev_skips = reversed(skips[:-1])
        for skip, (dec, upsample) in zip(rev_skips, zip(self.decoders[:-1], self.upsamples)):
            x = upsample(x)
            x = self.dropout(x)
            x = torch.cat((x, skip), dim=1)
            x = dec(x)

        x = self.decoders[-1](x)
        x = self.out_conv(x)
        return self.sigmoid(x)

def conv_block(self, in_channels, out_channels):
        """
        Creates a convolutional block with two convolutional layers, each followed by batch normalization and ReLU.

        Args:
            in_channels (int): Number of input channels for the block.
            out_channels (int): Number of output channels for the block.

        Returns:
            Sequential: A sequential container with convolutional layers, batch normalization, and ReLU activations.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

model = UNet(in_channels=3, out_channels=1)  # Assuming input is RGB and output is a binary mask
print(model)



model = UNet(in_channels=3, out_channels=1)  # Assuming input is RGB and output is a binary mask
print(model)
