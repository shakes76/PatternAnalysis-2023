import torch


class Model_Generator(torch.nn.Module):
    """
    convolutional super-resolution model

    """
    def __init__(self, num_channels=3):

        super().__init__()

        self._model = torch.nn.Sequential(

            self.make_layer(    # conv 1: input layer
                "input", num_channels, num_channels**2, batch_norm=True, activation_type="relu",
            ),
            self.make_layer(    # conv 2: intermediate layer
                "intermediate", num_channels**2, num_channels**2, batch_norm=True, activation_type="relu",
            ),

            self.make_layer(    # conv 3: upsampling layer (60 x 64 -> 120 x 128)
                "upsample", num_channels**2, num_channels**3, batch_norm=True, activation_type="relu",
            ),
            self.make_layer(    # conv 4: intermediate layer
                "intermediate", num_channels**3, num_channels**3, batch_norm=True, activation_type="relu",
            ),
            self.make_layer(    # conv 5: intermediate layer
                "intermediate", num_channels**3, num_channels**3, batch_norm=True, activation_type="relu",
            ),

            self.make_layer(    # conv 6: upsampling layer (120 x 128 -> 240 x 256)
                "upsample", num_channels**3, num_channels**2, batch_norm=True, activation_type="relu",
            ),
            self.make_layer(    # conv 7: intermediate layer
                "intermediate", num_channels**2, num_channels**2, batch_norm=True, activation_type="relu",
            ),

            self.make_layer(    # conv 8: output layer
                "output", num_channels**2, num_channels, batch_norm=False, activation_type="tanh",
            ),
        )

    def make_layer(
        self, layer_type, in_channels, out_channels, 
        batch_norm=True, activation_type="relu",
    ):
        """ 
        abstract layer method for creating convolutional layers for the generator

        """
        if layer_type == "upsample":
            layer = torch.nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1,
            )
        else:
            layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            )

        activation = torch.nn.ReLU() if activation_type == "relu" else torch.nn.Tanh()

        return torch.nn.Sequential(
            layer, torch.nn.BatchNorm2d(out_channels), activation,
        ) if batch_norm else torch.nn.Sequential(
            layer, activation,
        )

    def forward(self, x):
        """
        define the forward pass through the generator

        """
        return self._model(x)
    

class Model_Discriminator(torch.nn.Module):
    """
    convolutional super-resolution model

    """
    def __init__(self, num_channels=3):

        super().__init__()

        self._model = torch.nn.Sequential(

            self.make_layer(    # conv 1: input layer
                "input", num_channels, num_channels**2, batch_norm=False, activation_type="leaky-relu",
            ),

            self.make_layer(    # conv 2: downsampling layer (240 x 256 -> 120 x 128)
                "downsample", num_channels**2, num_channels**2, batch_norm=True, activation_type="leaky-relu",
            ),
            self.make_layer(    # conv 3: downsampling layer (120 x 128 -> 60 x 64)
                "downsample", num_channels**2, num_channels**2, batch_norm=True, activation_type="leaky-relu",
            ),
            self.make_layer(    # conv 4: downsampling layer (60 x 64 -> 30 x 32)
                "downsample", num_channels**2, num_channels**2, batch_norm=True, activation_type="leaky-relu",
            ),
            self.make_layer(    # conv 5: downsampling layer (30 x 32 -> 15 x 16)
                "downsample", num_channels**2, num_channels**2, batch_norm=True, activation_type="leaky-relu",
            ),

            self.make_layer(    # conv 6: output layer
                "output", num_channels**2, num_channels, batch_norm=True, activation_type="leaky-relu",
            ),

            # additional fully-connected output layers for classification
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=num_channels*15*16, out_features=1),
            torch.nn.Sigmoid(),
        )

    def make_layer(
        self, layer_type, in_channels, out_channels, 
        batch_norm=True, activation_type="relu",
    ):
        """
        abstract layer method for creating convolutional layers for the discriminator

        """
        if layer_type == "downsample":
            layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            )
        else:
            layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            )

        activation = torch.nn.LeakyReLU(0.2) if activation_type == "leaky-relu" else torch.nn.ReLU()

        return torch.nn.Sequential(
            layer, torch.nn.BatchNorm2d(out_channels), activation,
        ) if batch_norm else torch.nn.Sequential(
            layer, activation,
        )

    def forward(self, x):
        """
        define the forward pass through the discriminator

        """
        return self._model(x)
    