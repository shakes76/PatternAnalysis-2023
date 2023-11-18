import torch
import torch.nn as nn
import math
from modules import ResnetBlock, AttnBlock, Downsample, Upsample
from einops import reduce, repeat


class SkipOut(nn.Module):
    '''
        A module will record the last output and return identity of it.
        This module is used in out-edge of skip connection.
    '''

    def __init__(self):
        super(SkipOut, self).__init__()
        self.output = None

    def forward(self, x):
        self.output = x
        return x


class SkipIn(nn.Module):
    def __init__(self, skipOut: SkipOut):
        '''
            An module will add skipOut weight and add it into current neurons.
            This module is used in in-edge of skip connection.
        '''
        super(SkipIn, self).__init__()
        self.skipOut = skipOut

    def forward(self, x):
        return torch.cat([self.skipOut.output, x], dim=1)


class SkipConnect(nn.Module):
    '''
        An module will generate dict that record skipIn and skipOut module
        This module is used in skip connection.
    '''

    def __init__(self):
        super(SkipConnect, self).__init__()
        self.skipLayers = {}

    def registerSkip(self, key=None):
        # Register and Skipout and make corresponding Skipin.
        if key is None:
            key = len(self.skipLayers)
        skipOut = SkipOut()
        skipIn = SkipIn(skipOut)
        self.skipLayers[key] = [skipOut, skipIn]
        return skipOut

    def getSkip(self, idx):
        # Get skipin module.
        return self.skipLayers[idx][1]


class LatentDiffusionModel(nn.Module):
    def __init__(self, *, in_channels, ch, dropout=0.0, T=1000):

        super(LatentDiffusionModel, self).__init__()

        # Diffusion Parameters Setup
        self.T = T
        self.b = torch.linspace(1e-4, 0.02, T)
        self.a = 1 - self.b
        a_bar = []
        for t in range(T):
            a_bar.append((a_bar[-1] if t != 0 else 1) * self.a[t])
        self.a_bar = torch.Tensor(a_bar)

        # Time embedding Setup
        time_emb_size = ch
        time_embedding = torch.zeros([T, time_emb_size])
        div_term = torch.exp(torch.arange(0, time_emb_size, 2)
                             * (-math.log(10000.0) / time_emb_size))
        time_pos = torch.unsqueeze(torch.arange(0, T), -1)
        time_embedding[:, 0::2] = torch.sin(time_pos * div_term)
        time_embedding[:, 1::2] = torch.cos(time_pos * div_term)
        self.time_embedding = time_embedding

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.SiLU(),
            nn.Linear(time_emb_size, time_emb_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Embedding(32, ch),
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
        )

        # UNet Setup
        res_params = {
            'time_emb_size': time_emb_size,
            'dropout': dropout
        }

        skipConnect = SkipConnect()
        self.layers = nn.ModuleList([
            # Encoder Conv_in
            nn.Conv2d(in_channels, ch, 3, 1, 1),
            # Encoder Down 1
            ResnetBlock(in_channels=ch, **res_params),
            ResnetBlock(in_channels=ch, **res_params),
            AttnBlock(ch),
            skipConnect.registerSkip('L1'),
            # Encoder Down 2
            Downsample(ch, True),
            ResnetBlock(in_channels=ch, out_channels=ch * 2, **res_params),
            ResnetBlock(in_channels=ch * 2, **res_params),
            AttnBlock(ch * 2),
            skipConnect.registerSkip('L2'),
            # Encoder Mid
            ResnetBlock(in_channels=ch * 2, **res_params),
            AttnBlock(ch * 2),
            ResnetBlock(in_channels=ch * 2, **res_params),
            # Decoder Up 1
            skipConnect.getSkip('L2'),
            ResnetBlock(in_channels=ch * 2 * 2,
                        out_channels=ch * 2, **res_params),
            ResnetBlock(in_channels=ch * 2, **res_params),
            AttnBlock(ch * 2),
            Upsample(ch * 2, with_conv=True),
            # Decoder Up 1
            skipConnect.getSkip('L1'),
            ResnetBlock(in_channels=ch * 3, out_channels=ch, **res_params),
            ResnetBlock(in_channels=ch, **res_params),
            AttnBlock(ch),
            # Decoder End
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, 1, 1)
        ])

    def get_noise(self, x, t):
        # Generate noisy image with a specific weight.
        # What weight should be used can refer to tech_node.md
        device = next(self.parameters()).device
        noise = torch.randn_like(x, device=device)
        weight = self.a_bar[t][:, None, None, None].to(device)
        noise_image = weight**0.5 * x + (1 - weight)**0.5 * noise
        return noise_image, noise

    def forward(self, x, t, cond):
        # Get time embedding
        self.time_embedding = self.time_embedding.to(t.device)
        time_emb = self.time_embedding[t]
        time_emb = self.time_mlp(time_emb)
        # Get condition embedding
        cond_emb = self.cond_mlp(cond)

        h = x
        for layer in self.layers:
            if isinstance(layer, ResnetBlock):
                # Merge two condition (time in diffusion and z-index condition) into one.
                h = layer(h, time_encode=time_emb + cond_emb)
            else:
                h = layer(h)
        return h

    def sample_with_cond(self, shape, cond, keep_mid=False):
        # Sample images with condition.
        # keep_mid: True will return all the images into one list through inference process.

        # Get deivce
        device = next(self.parameters()).device
        # batch_size = cond shape. (will generate shape[0] * batch_size images.)
        batch_size = cond.shape[0]

        # Decide the initial noise
        cur_xt = torch.randn(shape, device=device)
        # Repeat "cond len" times for initial noise.
        cur_xt = repeat(cur_xt, 'n c h w -> (n repeat) c h w',
                        repeat=batch_size)

        # Return a list if keep_mid is set.
        if keep_mid:
            ret = [cur_xt.cpu()]

        # Denoising process in DDPM
        for t in range(self.T-1, -1, -1):
            # Generate t-time
            batch_t = torch.zeros(size=(batch_size, ),
                                  device=device, dtype=torch.long) + t
            predicted_noise = self(cur_xt, batch_t, cond)

            # Reconstruct the original image with specific weights.
            next_xt = self.a[t] ** -0.5 * (cur_xt - (1 - self.a[t]) / (
                (1 - self.a_bar[t]) ** 0.5) * predicted_noise)

            # If the denoise process is not end, we will add a new noise into image.
            # !!! It's an important note that this noise is crucial and can't be ignored.
            if t != 0:
                sigma = self.b[t] ** 0.5
                next_xt += torch.randn(shape, device=device) * sigma

            cur_xt = next_xt

            # Normalize the output into 0~1.
            # !!! You can output without normalization.
            min_im = reduce(cur_xt, 'b c h w -> b 1 1 1', 'min')
            max_im = reduce(cur_xt, 'b c h w -> b 1 1 1', 'max')
            out_xt = (cur_xt - min_im) / (max_im - min_im)
            out_xt = cur_xt.detach()

            # Record the intermediate value if keep_mid.
            if keep_mid:
                ret.append(out_xt.cpu())

        if keep_mid:
            return ret
        else:
            return out_xt
