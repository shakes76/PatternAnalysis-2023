# reference: https://huggingface.co/spaces/danurahul/gan/blob/main/taming-transformers/taming/modules/discriminator/model.py
import torch
import torch.nn as nn
from modules import AttnBlock

from util import sinusoidal_embedding


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, attn=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminator, self).__init__()

        # Time embedding Setup
        # Time embedding in this funtion is the z-index of brain.
        # And this condition is conditional GAN.
        time_emb_size = 32
        self.time_embedding = sinusoidal_embedding(
            pos_len=32, time_emb_size=time_emb_size)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.SiLU(),
            nn.Linear(time_emb_size, time_emb_size),
        )

        kw = 4
        padw = 1

        # First Layers: an classic conv2d
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ndf, kernel_size=kw,
                          stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            )
        ])
        nf_mult = 1
        nf_mult_prev = 1

        # Mid Layers: gradually increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.convs += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.GroupNorm(ndf//4, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.convs += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.GroupNorm(ndf//4, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Final Layers: predict true or false in one channel.
        self.conv_out = nn.Conv2d(
            ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        # Build time embedding block.
        # Time embedding will added after convolution in mid layers
        self.time_emb = nn.ModuleList([])
        for conv in self.convs:
            if isinstance(conv, nn.Conv2d):
                out_channels = conv.weight.shape[0]
                self.time_emb.append(nn.Sequential(
                    nn.Linear(time_emb_size, out_channels),
                    nn.SiLU(),
                ))

        # Should we add attention in discriminator
        if attn:
            self.attn = AttnBlock(8 * ndf)
        else:
            self.attn = None

    def forward(self, input, t):
        """Standard forward."""
        self.time_embedding = self.time_embedding.to(t.device)
        time_emb = self.time_embedding[t]
        time_emb = self.time_mlp(time_emb)

        h = input
        cnt = 0
        for conv in self.convs:
            h = conv(h)
            # Put every time emb after conv2d.
            # Note that first conv will not be included because it's encapsuled by nn.Sequential
            if isinstance(conv, nn.Conv2d):
                h = h + self.time_emb[cnt](time_emb)[:, :, None, None]
                cnt += 1

        # Only if attn block is defined we pass through attn block
        if self.attn:
            h = self.attn(h)

        h = self.conv_out(h)
        return h
