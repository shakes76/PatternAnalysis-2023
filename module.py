import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, time_emb_size=512):
        super().__init__()

        # Record useful arguments
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # Define Resblocks 1
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )

        # Define Time Embedding
        if time_emb_size > 0:
            self.time_emb = nn.Sequential(
                nn.Linear(time_emb_size, out_channels),
                nn.SiLU(),
            )

        # Define Resblocks 2
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        # Define shortcut
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, time_encode):
        h = x

        # Pass block 1
        h = self.block1(h)
        # Add time embedding / encoding
        if time_encode is not None:
            h = h + self.time_emb(time_encode)[:, :, None, None]
        # Pass block 2
        h = self.block2(h)
        # If size not match, add shortcut.
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)

        return x+h
