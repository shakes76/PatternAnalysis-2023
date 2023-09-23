from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Self defined modules
from module import ResnetBlock, AttnBlock, Downsample, Upsample


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignores):
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


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 resolution, z_channels, **ignores):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Calculate block_in & current resolution
        block_in = ch*ch_mult[-1]
        curr_res = resolution // 2**(self.num_resolutions-1)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, dropout=dropout)
        )

        # upsampling
        self.up = nn.Sequential()

        # Travel in reversed order.
        for i_level in reversed(range(self.num_resolutions)):
            blocks = nn.Sequential()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                blocks.append(ResnetBlock(in_channels=block_in,
                                          out_channels=block_out,
                                          dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out
            if i_level != 0:
                blocks.append(Upsample(block_in, resamp_with_conv))
                curr_res = curr_res * 2
            self.up.append(blocks)

        # end
        self.end = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, 3, 1, 1)
        )

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid(h)

        # upsampling
        h = self.up(h)

        # end
        h = self.end(h)
        return h


class Autoencoder(nn.Module):
    def __init__(self, *, ch=64, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, resolution=256, z_channels=16, embed_dim=16):
        super().__init__()
        params = {
            'ch': ch,
            'in_channels': 1,
            'out_ch': 1,
            'ch_mult': ch_mult,
            'num_res_blocks': num_res_blocks,
            'attn_resolutions': attn_resolutions,
            'dropout': dropout,
            'resamp_with_conv': resamp_with_conv,
            'resolution': resolution,
            'z_channels': z_channels
        }
        self.encoder = Encoder(**params)
        self.conv_before_reparm = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.conv_after_reparm = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(**params)

        # For convienient purpose, store the z-shape.
        latent_wh = resolution // 2 ** (len(ch_mult)-1)
        self.z_shape = [z_channels, latent_wh, latent_wh]

    def encode(self, x):
        h = self.encoder(x)
        h = self.conv_before_reparm(h)
        return h

    def decode(self, z):
        h = self.conv_after_reparm(z)
        h = self.decoder(h)
        return h

    def sample(self, batch_size):
        # Get current device
        device = next(self.parameters()).device
        # Generate z space from randn.
        z = torch.randn([batch_size, *self.z_shape]).to(device=device)
        return self.decode(z)

    def reparameterization(self, z):
        # The detail of reparameterization trick is documented in README.md

        # Split two parts: mean and std
        z_mean, z_logvar = torch.chunk(z, 2, dim=1)
        # Clamp z_logvar to avoid strange problem.
        z_logvar = torch.clamp(z_logvar, -30.0, 20.0)
        z_std = torch.exp(0.5 * z_logvar)
        z_var = torch.exp(z_logvar)
        z = z_mean + z_std * torch.randn_like(z_mean).to(device=z_mean.device)

        kl_dis = torch.pow(z_mean, 2) + z_var - 1.0 - z_logvar
        kl_dis = 0.5 * torch.sum(kl_dis, dim=[1, 2, 3])

        return z, kl_dis

    def forward(self, x):
        latent = self.encode(x)
        z, kl_dis = self.reparameterization(latent)
        recon = self.decode(z)
        return recon, latent, kl_dis


if __name__ == '__main__':
    # net = Encoder(double_z=True, z_channels=16, resolution=256, in_channels=1, ch=64, ch_mult=[
    #     1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).cuda()

    # summary(net, input_size=(16, 1, 256, 256))
    # print(net(torch.randn([16, 1, 256, 256]).cuda()).shape)

    # net = Decoder(z_channels=16, resolution=256, ch=64, out_ch=1, ch_mult=[
    #             1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).cuda()

    # summary(net, input_size=(16, 16, 16, 16))
    # print(net(torch.randn([16, 16, 16, 16]).cuda()).shape)

    net = Autoencoder().cuda()
    summary(net, input_size=(16, 1, 256, 256))
    print([out.shape for out in net(torch.randn([16, 1, 256, 256]).cuda())])
