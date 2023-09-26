import torch


class Model(torch.nn.Module):
    """
    convolutional super-resolution model

    """
    def __init__(self, num_channels=3):

        super().__init__()

        self._model = torch.nn.Sequential(

            # input layer
            torch.nn.Conv2d(
                in_channels=num_channels, out_channels=num_channels**2,
                kernel_size=3, stride=1, padding=1,
            ),
            torch.nn.ReLU(),

            # upsampling layer: 60 x 64 -> 120 x 128
            torch.nn.ConvTranspose2d(
                in_channels=num_channels**2, out_channels=num_channels**3,
                kernel_size=4, stride=2, padding=1,
            ),
            torch.nn.ReLU(),

            # intermediate layer
            torch.nn.Conv2d(
                in_channels=num_channels**3, out_channels=num_channels**3,
                kernel_size=3, stride=1, padding=1,
            ),
            torch.nn.ReLU(),

            # upsampling layer: 120 x 128 -> 240 x 256
            torch.nn.ConvTranspose2d(
                in_channels=num_channels**3, out_channels=num_channels**2,
                kernel_size=4, stride=2, padding=1,
            ),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Conv2d(
                in_channels=num_channels**2, out_channels=num_channels,
                kernel_size=3, stride=1, padding=1,
            ),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self._model(x)
