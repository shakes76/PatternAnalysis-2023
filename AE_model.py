from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Self defined modules
from module import ResnetBlock, AttnBlock, Downsample


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True):
        '''
            ch: base number of all channels, int
            ch_mult: channal number of each blocks, [int...]
            num_res_blocks: numbers of resblocks in each block, int.
            attn_resolutions: Do self-attention when current resolution is in attn_resolutions, [int...]
            dropout: droupout ratio in resblock, float ~ [0, 1]
            resamp_with_conv: Do conv(strides=2) if True, else we do avgpool(strides=2)
            in_channels: input_channels, int
            resolution: image's width and height, int (width should be equal to height)
            z_channels: latent size will be [z_channels, resolution/2**len(ch_mult), ..], int
            double_z: should this model output double size of z to implement reparametrization.
        '''
        super().__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # First Conv
        self.conv_in = torch.nn.Conv2d(in_channels, ch, 3, 1, 1)

        # Build sequential of down list
        self.down = nn.Sequential()
        block_in = ch
        curr_res = resolution
        for i_level in range(self.num_resolutions):
            # Calculate in & out
            block_out = ch_mult[i_level] * ch

            blocks = nn.Sequential()

            for i_block in range(self.num_res_blocks):

                # Resblock like Unet
                blocks.append(ResnetBlock(in_channels=block_in,
                              out_channels=block_out, dropout=dropout))

                # Attention Block (Only add when resolution is low enough.)
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            # Downsample block (Until reach the target resolutions)
            if i_level != self.num_resolutions-1:
                blocks.append(Downsample(block_in, resamp_with_conv))
                curr_res = curr_res // 2
            self.down.append(blocks)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, dropout=dropout)
        )

        # end
        self.end = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, z_channels * (2 if double_z else 1), 3, 1, 1)
        )

    def forward(self, x):

        # first conv
        h = self.conv_in(x)

        # down conv
        h = self.down(h)

        # middle
        h = self.mid(h)

        # end
        h = self.end(h)
        return h


if __name__ == '__main__':
    net = Encoder(double_z=True, z_channels=16, resolution=256, in_channels=1, ch=64, ch_mult=[
        1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).cuda()

    summary(net, input_size=(16, 1, 256, 256))
    print(net(torch.randn([16, 1, 256, 256]).cuda()).shape)
