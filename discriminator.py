# reference: https://huggingface.co/spaces/danurahul/gan/blob/main/taming-transformers/taming/modules/discriminator/model.py
import torch
import torch.nn as nn
from module import AttnBlock

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

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ndf, kernel_size=kw,
                          stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            )
        ])
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=False),
                    # nn.BatchNorm2d(ndf * nf_mult),
                    nn.GroupNorm(32, ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=1, padding=padw, bias=False),
                # nn.BatchNorm2d(ndf * nf_mult),
                nn.GroupNorm(32, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            )
        )
        self.conv_out = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        # Layers for LPIPS
        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(),
                # conv: [Conv2d, BN, LeakyReLU]
                nn.Conv2d(conv[0].weight.shape[0], 1, 1, stride=1, padding=0, bias=False)
            )
            for conv in self.convs
        ])
        
        self.attn = AttnBlock(512)

    def forward(self, input):
        """Standard forward."""
        h = input
        for conv in self.convs:
            h = conv(h)
        h = self.attn(h)
        h = self.conv_out(h)
        return h
    
    def LPIPS(self, input1, input2):
        h1, h2 = input1, input2
        score = None
        for conv, lin in zip(self.convs, self.lins):
            # In each layer, calculate score by proj(square-error of two images)
            h1, h2 = conv(h1), conv(h2)
            feats1, feats2 = normalize_tensor(h1), normalize_tensor(h2)
            diff = spatial_average(lin( (feats1 - feats2) ** 2), keepdim=True)

            if score is None:
                score = diff
            else:
                score += diff

        return score        

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


if __name__ == '__main__':
    discriminator = NLayerDiscriminator(
        input_nc=1, n_layers=3).apply(weights_init).cuda()
    from torchinfo import summary
    summary(discriminator, (4, 1, 256, 256))
    X = torch.randn([16, 1, 256, 256]).cuda()
 
    print(discriminator(X).shape)
    LPIPS_score = discriminator.LPIPS(X, X)
    print((X + LPIPS_score).shape)
