import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self, n_classes):
        super(UNET, self).__init__()

        # Define the encoding layers
        self.encode1 = self.conv_block(3, 64)
        self.encode2 = self.conv_block(64, 128)
        self.encode3 = self.conv_block(128, 256)
        self.encode4 = self.conv_block(256, 512)
        self.encode5 = self.conv_block(512, 1024)

        # Define the decoding layers with upconvolutions
        self.decode1 = self.upconv_block(1024, 512)
        self.decode2 = self.upconv_block(512, 256)
        self.decode3 = self.upconv_block(256, 128)
        self.decode4 = self.upconv_block(128, 64)

        # Define additional convolution layers to manage channel sizes after concatenation
        self.decode_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decode_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decode_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decode_conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Final output layer
        self.decode_output = nn.Conv2d(64, n_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def upconv_and_concat(self, encode_feature, decode_feature, channels):
        upconv_layer = self.upconv_block(channels*2, channels)
        upconv = upconv_layer(decode_feature)
        return torch.cat([upconv, encode_feature], dim=1)

    def forward(self, x):

        # Encode path
        encode1 = self.encode1(x)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        encode5 = self.encode5(encode4)

        # Decode path
        decode1_input = self.upconv_and_concat(encode4, encode5, 512)
        decode1 = self.decode_conv1(decode1_input)

        decode2_input = self.upconv_and_concat(encode3, decode1, 256)
        decode2 = self.decode_conv2(decode2_input)

        decode3_input = self.upconv_and_concat(encode2, decode2, 128)
        decode3 = self.decode_conv3(decode3_input)

        decode4_input = self.upconv_and_concat(encode1, decode3, 64)
        decode4 = self.decode_conv4(decode4_input)

        output = self.decode_output(decode4)

        return torch.sigmoid(output)


