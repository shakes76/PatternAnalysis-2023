from torchinfo import summary
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler

# Self defined modules
from modules import ResnetBlock, AttnBlock, Downsample, Upsample, VectorQuantizer2
from util import sinusoidal_embedding


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, pos_len=32, **ignores):
        '''
            Define encoder for autoencoder.

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

        # Time embedding Setup
        time_emb_size = ch
        self.time_embedding = sinusoidal_embedding(pos_len, time_emb_size)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.SiLU(),
            nn.Linear(time_emb_size, time_emb_size),
        )

        # Setup
        res_params = {
            'time_emb_size': time_emb_size,
            'dropout': dropout
        }

        self.layers = nn.ModuleList([])
        # First Conv
        self.layers.append(torch.nn.Conv2d(in_channels, ch, 3, 1, 1))

        # Build sequential of down list
        block_in = ch
        curr_res = resolution
        for i_level in range(self.num_resolutions):
            # Calculate in & out
            block_out = ch_mult[i_level] * ch

            for i_block in range(self.num_res_blocks):

                # Resblock like Unet
                self.layers.append(ResnetBlock(in_channels=block_in,
                                               out_channels=block_out, **res_params))

                # Attention Block (Only add when resolution is low enough.)
                if curr_res in attn_resolutions:
                    self.layers.append(AttnBlock(block_out))
                block_in = block_out

            # Downsample block (Until reach the target resolutions)
            if i_level != self.num_resolutions-1:
                self.layers.append(Downsample(block_in, resamp_with_conv))
                curr_res = curr_res // 2

        # middle
        self.layers += ([
            # Middle Part
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, **res_params),
            AttnBlock(block_in),
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, **res_params),
            # End Part
            nn.GroupNorm(num_groups=32, num_channels=block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, z_channels * (2 if double_z else 1), 3, 1, 1)
        ])

    def forward(self, x, t):
        # Get time embedding
        self.time_embedding = self.time_embedding.to(t.device)
        time_emb = self.time_embedding[t]
        time_emb = self.time_mlp(time_emb)

        h = x
        for layer in self.layers:
            if isinstance(layer, ResnetBlock):
                h = layer(h, time_encode=time_emb)
            else:
                h = layer(h)
        return h

    def get_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.layers[-1].weight


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 resolution, z_channels, tanh_out=False, pos_len=32, **ignores):
        '''
            Define decoder for autoencoder.
            Parameters setting please refer to Encoder.
        '''
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        # Time embedding Setup
        time_emb_size = ch
        self.time_embedding = sinusoidal_embedding(pos_len, time_emb_size)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.SiLU(),
            nn.Linear(time_emb_size, time_emb_size),
        )

        # Setup
        res_params = {
            'time_emb_size': time_emb_size,
            'dropout': dropout
        }

        # Calculate block_in & current resolution
        block_in = ch*ch_mult[-1]
        curr_res = resolution // 2**(self.num_resolutions-1)

        self.layers = nn.ModuleList([])

        # middle
        self.layers += [
            # Conv in
            torch.nn.Conv2d(z_channels, block_in, 3, 1, 1),
            # Middle part
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, **res_params),
            AttnBlock(block_in),
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in, **res_params)
        ]

        # Upsampling, Travel in reversed order.
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                self.layers.append(ResnetBlock(in_channels=block_in,
                                               out_channels=block_out, **res_params))
                if curr_res in attn_resolutions:
                    self.layers.append(AttnBlock(block_out))
                block_in = block_out
            if i_level != 0:
                self.layers.append(Upsample(block_in, resamp_with_conv))
                curr_res = curr_res * 2

        # End part
        self.layers += [
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, 3, 1, 1)
        ]

    def forward(self, x, t):
        # Get time embedding
        self.time_embedding = self.time_embedding.to(t.device)
        time_emb = self.time_embedding[t]
        time_emb = self.time_mlp(time_emb)

        h = x
        for layer in self.layers:
            if isinstance(layer, ResnetBlock):
                h = layer(h, time_encode=time_emb)
            else:
                h = layer(h)

        if self.tanh_out:
            h = torch.tanh(h)
        return h

    def get_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.layers[-1].weight


class VAE(nn.Module):
    def __init__(self, *, ch=64, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, resolution=256, z_channels=16, embed_dim=16, pos_len=32):
        '''
            Define vanilla VAE model with reparametrization trick. 
            Parameters setting please refer to Encoder.
        '''
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
            'z_channels': z_channels,
            'pos_len': pos_len
        }
        self.encoder = Encoder(**params)
        self.conv_before_reparm = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.conv_after_reparm = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(tanh_out=True, **params)
        self.pos_len = pos_len
        # For convienient purpose, store the z-shape.
        latent_wh = resolution // 2 ** (len(ch_mult)-1)
        self.z_shape = [embed_dim, latent_wh, latent_wh]

    def encode(self, x, t):
        h = self.encoder(x, t)
        h = self.conv_before_reparm(h)
        return h

    def decode(self, z, t):
        h = self.conv_after_reparm(z)
        h = self.decoder(h, t)
        return h

    def sample(self, batch_size, t):
        # Get current device
        device = next(self.parameters()).device
        # Generate z space from randn.
        z = torch.randn([batch_size, *self.z_shape]).to(device=device)
        return self.decode(z, t)

    def sample_for_visualize(self, batch_size):
        '''
            This will output 32 (pos_len) images.
        '''
        # Get current device
        device = next(self.parameters()).device
        # Generate z space from randn.
        z = torch.randn([1, *self.z_shape]).to(device=device)
        z = repeat(z, 'b z h w -> (repeat b) z h w', repeat=self.pos_len)
        t = torch.arange(0, self.pos_len, device=device, dtype=torch.long)
        # out: will contain pos_len images
        out = []
        for i in range(0, self.pos_len, batch_size):
            start, end = i, min(self.pos_len, i+batch_size)
            out.append(self.decode(z[start:end], t[start:end]))
        out = torch.cat(out)
        return out

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

    def forward(self, x, t):
        latent = self.encode(x, t)
        z, kl_dis = self.reparameterization(latent)
        recon = self.decode(z, t)
        return recon, kl_dis, latent

    def get_decoder_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.decoder.get_last_layer()

    def get_encoder_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.encoder.get_last_layer()


class VQVAE(nn.Module):
    def __init__(self, *, ch=64, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, resolution=256, z_channels=16, embed_dim=8, n_embed=256, pos_len=32):
        '''
            Define VQVAE model. 
            Parameters setting please refer to Encoder.
        '''
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
            'z_channels': z_channels,
            'pos_len': pos_len
        }
        self.n_embed = n_embed
        self.encoder = Encoder(**params, double_z=False)
        self.conv_before_quant = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.conv_after_quant = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(tanh_out=True, **params)
        self.pos_len = pos_len

        self.quantize = VectorQuantizer2(
            n_e=n_embed, e_dim=embed_dim, beta=0.25)
        # For convienient purpose, store the z-shape.
        latent_wh = resolution // 2 ** (len(ch_mult)-1)
        self.z_shape = [latent_wh, latent_wh]
        # Weight sampler is just a sampler and shouldn't be backprop
        self.weight_sampler = torch.ones(self.z_shape + [n_embed]).detach()

    def encode(self, x, t):
        h = self.encoder(x, t)
        h = self.conv_before_quant(h)
        return h

    def decode(self, z, t):
        h = self.conv_after_quant(z)
        h = self.decoder(h, t)
        return h

    def sample(self, batch_size, t):
        # Get current device
        device = next(self.parameters()).device
        # Generate z space from w random sampler.
        with torch.no_grad():
            w, h, n_embed = self.weight_sampler.shape
            # Change indices record into S, nE shape
            weight_sampler = rearrange(
                self.weight_sampler, 'h w nE -> (h w) nE')
            # Weighted sample from distribution
            ind = torch.tensor(list(WeightedRandomSampler(
                weight_sampler, num_samples=batch_size))).detach()
            # Reshape into correct type
            ind = rearrange(ind, '(h w) nE -> h w nE', w=w)
            # To correct device.
            # Note that there're some issue using WeightedRandomSampler in GPU)
            ind = ind.to(device)
        z_q = self.quantize.embedding(ind)
        z_q = rearrange(z_q, 'h w b c -> b c h w')
        return self.decode(z_q, t)

    def sample_for_visualize(self, batch_size):
        # Get current device
        # Generate z space from w random sampler.
        device = next(self.parameters()).device

        with torch.no_grad():
            # This part is same as sample. Please check the above funciton.
            w, h, n_embed = self.weight_sampler.shape
            weight_sampler = rearrange(
                self.weight_sampler, 'h w nE -> (h w) nE')
            ind = torch.tensor(list(WeightedRandomSampler(
                weight_sampler, num_samples=1))).detach()
            ind = rearrange(ind, '(h w) nE -> h w nE', w=w)
            ind = ind.to(device)
        z_q = self.quantize.embedding(ind)
        z_q = rearrange(z_q, 'h w b c -> b c h w')

        z_q = repeat(z_q, 'b c h w -> (repeat b) c h w', repeat=self.pos_len)
        # Should sample z_index from 0 to self.pos_len (32 in OASIS dataset)
        t = torch.arange(0, self.pos_len, device=device, dtype=torch.long)

        return self.decode(z_q, t)

    def forward(self, x, t):
        latent = self.encode(x, t)
        quant, diff_loss, ind = self.quantize(latent)
        recon = self.decode(quant, t)
        return recon, diff_loss, ind

    def get_decoder_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.decoder.get_last_layer()

    def get_encoder_last_layer(self):
        # This function is for adpative loss for discriminator.
        return self.encoder.get_last_layer()

    def update_sampler(self, weight_sampler):
        # Update weight sampler for weighted sampling
        self.weight_sampler = weight_sampler.detach()
