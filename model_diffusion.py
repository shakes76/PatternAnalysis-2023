import torch
import torch.nn as nn
import math
import numpy as np
from module import ResnetBlock, AttnBlock, Downsample, Upsample, VectorQuantizer2
from einops import reduce
class SkipOut(nn.Module):
    def __init__(self):
        super(SkipOut, self).__init__()
        self.output = None    

    def forward(self, x):
        self.output = x
        return x

class SkipIn(nn.Module):
    def __init__(self, skipOut: SkipOut):
        super(SkipIn, self).__init__()
        self.skipOut = skipOut
    def forward(self, x):
        return torch.cat([self.skipOut.output, x], dim=1)

class SkipConnect(nn.Module):
    def __init__(self):
        super(SkipConnect, self).__init__()
        self.skipLayers = {}

    def registerSkip(self, key=None):
        if key is None:
            key = len(self.skipLayers)
        skipOut = SkipOut()
        skipIn = SkipIn(skipOut)
        self.skipLayers[key] = [skipOut, skipIn]
        return skipOut

    def getSkip(self, idx):
        return self.skipLayers[idx][1]
class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding
class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class LatentDiffusionModel(nn.Module):
    def __init__(self, *, in_channels, ch, dropout=0.0, T=1000):

        super(LatentDiffusionModel, self).__init__()

        # Diffusion Parameters Setup
        self.T = T
        self.b = torch.linspace(1e-4, 0.02, T)
        self.a = 1 - self.b
        a_bar = []
        for t in range(T):
            a_bar.append( (a_bar[-1] if t != 0 else 1) * self.a[t])
        self.a_bar = torch.Tensor(a_bar)

        # Time embedding Setup
        time_emb_size = ch
        time_embedding = torch.zeros([T, time_emb_size])
        div_term = torch.exp(torch.arange(0, time_emb_size, 2) * (-math.log(10000.0) / time_emb_size))
        time_pos = torch.unsqueeze(torch.arange(0, T), -1)
        time_embedding[:, 0::2] = torch.sin(time_pos * div_term)
        time_embedding[:, 1::2] = torch.cos(time_pos * div_term)
        self.time_embedding = time_embedding
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_size, time_emb_size),
            nn.SiLU(),
            nn.Linear(time_emb_size, time_emb_size),
        ) 

        # UNet Setup
        res_params = {
            'time_emb_size' : time_emb_size,
            'dropout' : dropout
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
            ResnetBlock(in_channels=ch, out_channels = ch * 2, **res_params),
            ResnetBlock(in_channels=ch * 2, **res_params),
            AttnBlock(ch * 2),
            skipConnect.registerSkip('L2'),
            # Encoder Mid
            ResnetBlock(in_channels=ch * 2, **res_params),
            AttnBlock(ch * 2),
            ResnetBlock(in_channels=ch * 2, **res_params),
            # Decoder Up 1
            skipConnect.getSkip('L2'),
            ResnetBlock(in_channels=ch * 2 * 2, out_channels = ch * 2, **res_params),
            ResnetBlock(in_channels=ch * 2, **res_params),
            AttnBlock(ch * 2),
            Upsample(ch * 2, with_conv=True),
            # Decoder Up 1
            skipConnect.getSkip('L1'),
            ResnetBlock(in_channels=ch * 3, out_channels = ch, **res_params),
            ResnetBlock(in_channels=ch, **res_params),
            AttnBlock(ch),
            # Decoder End
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, 1, 1)
        ])

    def get_noise(self, x, t):
        device = next(self.parameters()).device
        noise = torch.randn_like(x, device=device)
        weight = self.a_bar[t][:, None, None, None].to(device)
        noise_image = weight**0.5 * x + (1 - weight)**0.5 * noise
        return noise_image, noise

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
    
    def sample(self, shape, keep_mid=False):
        device = next(self.parameters()).device
        batch_size = shape[0]
        cur_xt = torch.randn(shape, device=device)
        if keep_mid:
            ret = [cur_xt.cpu()]
        for t in range(self.T-1, -1, -1):
            batch_t = torch.zeros(size=(batch_size, ), device=device, dtype=torch.long) + t
            predicted_noise = self(cur_xt, batch_t)
            next_xt = self.a[t] ** -0.5 * (cur_xt - (1 - self.a[t]) / ((1 - self.a_bar[t]) ** 0.5) * predicted_noise)
            if t != 0:
                sigma = self.b[t] ** 0.5
                next_xt += torch.randn(shape, device=device) * sigma

            cur_xt = next_xt

            if t == 0 or keep_mid:
                # Normalize the output lest the image be scaled.
                min_im = reduce(cur_xt, 'b c h w -> b 1 1 1', 'min')
                max_im = reduce(cur_xt, 'b c h w -> b 1 1 1', 'max')
                out_xt = (cur_xt - min_im) / (max_im - min_im)
            if keep_mid:
                ret.append(out_xt.cpu())
        if keep_mid:
            return ret
        else:
            return out_xt

if __name__ == '__main__':
    net = LatentDiffusionModel(in_channels=8, ch=32).cuda()
    from torchinfo import summary
    # print(summary(net, input_size=(2, 8, 16, 16)))
    x = torch.randn((2, 8, 16, 16)).cuda()
    t = torch.randint(low=1, high=1000, size=(2,)).cuda()
    print(summary(net, input_data=(x, t)))
    net(x, t)
    # with torch.no_grad():
    #     net.sample((64, 1, 28, 28), True)
    # optimizer = torch.optim.Adam(net.parameters(), 1e-4)
    # optimizer.zero_grad()
    # loss = net(torch.Tensor([[5]]).cuda())
    # loss.backward()
    # optimizer.step()
    # from dataset import MyDataset
    # dataset = MyDataset('test', limit=32)
    # import matplotlib.pyplot as plt
    # m = dataset[0][0][0][:, :, None] * 2 - 1
    # import numpy as np
    # for i in range(1000):
    #     noise = np.random.random(m.shape)
    #     m += noise
    #     if i % 50 == 0:
    #         plt.imshow(m, cmap='gray')
    #         plt.show()
    # print()
    