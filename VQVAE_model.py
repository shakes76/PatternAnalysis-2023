from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
# Self defined modules
from module import ResnetBlock, AttnBlock, Downsample, Upsample, VectorQuantizer2
from torch.utils.data import WeightedRandomSampler

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

    def get_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.end[-1].weight

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 resolution, z_channels, tanh_out=False, **ignores):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

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

        if self.tanh_out:
            h = torch.tanh(h)
        return h
    
    def get_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.end[-1].weight


class VQVAE(nn.Module):
    def __init__(self, *, ch=64, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, resolution=256, z_channels=16, embed_dim=8, n_embed=256):
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
        self.n_embed = n_embed
        self.encoder = Encoder(**params, double_z=False)
        self.conv_before_quant = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.conv_after_quant = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(tanh_out=True, **params)

        self.quantize = VectorQuantizer2(n_e=n_embed, e_dim=embed_dim, beta=0.25,
                                        remap=None,
                                        sane_index_shape=False)
        # For convienient purpose, store the z-shape.
        latent_wh = resolution // 2 ** (len(ch_mult)-1)
        self.z_shape = [latent_wh, latent_wh]
        # Weight sampler is just a sampler and shouldn't be backprop
        self.weight_sampler = torch.ones(self.z_shape + [n_embed]).detach()

    def encode(self, x):
        h = self.encoder(x)
        h = self.conv_before_quant(h)
        return h

    def decode(self, z):
        h = self.conv_after_quant(z)
        h = self.decoder(h)
        return h

    def sample(self, batch_size):
        # Get current device
        # Generate z space from randn.
        DEVICE = next(self.parameters()).device

        with torch.no_grad():
            w, h, n_embed = self.weight_sampler.shape
            weight_sampler = rearrange(self.weight_sampler, 'w h nE -> (w h) nE')
            ind = torch.tensor(list(WeightedRandomSampler(weight_sampler, num_samples=batch_size))).detach()
            ind = rearrange(ind, '(w h) nE -> w h nE', w=w)
            ind = ind.to(DEVICE)
        z_q = self.quantize.embedding(ind)
        z_q = rearrange(z_q, 'w h b c -> b c h w')
        return self.decode(z_q)

    def forward(self, x):
        latent = self.encode(x)
        quant, diff_loss, (_, _, ind) = self.quantize(latent)
        recon = self.decode(quant)
        return recon, diff_loss, ind

    def get_decoder_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.decoder.get_last_layer()

    def get_encoder_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.encoder.get_last_layer()
    
    def update_sampler(self, weight_sampler):
        self.weight_sampler = weight_sampler.detach()

if __name__ == '__main__':
    # net = Encoder(double_z=True, z_channels=16, resolution=256, in_channels=1, ch=64, ch_mult=[
    #     1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).cuda()

    # summary(net, input_size=(16, 1, 256, 256))
    # print(net(torch.randn([16, 1, 256, 256]).cuda()).shape)

    # net = Decoder(z_channels=16, resolution=256, ch=64, out_ch=1, ch_mult=[
    #             1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0).cuda()

    # summary(net, input_size=(16, 16, 16, 16))
    # print(net(torch.randn([16, 16, 16, 16]).cuda()).shape)

    net = VQVAE().cuda()
    summary(net, input_size=(16, 1, 256, 256))
    X = torch.randn([16, 1, 256, 256]).cuda()
    print([out.shape for out in net(X)])
    net.sample(16)
    # print(net.get_last_layer().shape)