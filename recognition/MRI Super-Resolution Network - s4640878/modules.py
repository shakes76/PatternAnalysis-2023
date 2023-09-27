import torch


class Model_Generator(torch.nn.Module):
    """
    convolutional super-resolution model

    """
    def __init__(self, num_channels=3):

        super().__init__()

        self._model = torch.nn.Sequential(

            # input layer
            self.make_layer(
                "input", num_channels, num_channels**2, 
                batch_norm=True, activation_type="relu",
            ),

            # intermediate layer
            self.make_layer(
                "intermediate", num_channels**2, num_channels**2, 
                batch_norm=True, activation_type="relu",
            ),

            # upsampling layer: 60 x 64 -> 120 x 128
            self.make_layer(
                "upsample", num_channels**2, num_channels**3, 
                batch_norm=True, activation_type="relu",
            ),

            # intermediate layer
            self.make_layer(
                "intermediate", num_channels**3, num_channels**3, 
                batch_norm=True, activation_type="relu",
            ),

            # intermediate layer
            self.make_layer(
                "intermediate", num_channels**3, num_channels**3, 
                batch_norm=True, activation_type="relu",
            ),

            # upsampling layer: 120 x 128 -> 240 x 256
            self.make_layer(
                "upsample", num_channels**3, num_channels**2, 
                batch_norm=True, activation_type="relu",
            ),

            # intermediate layer
            self.make_layer(
                "intermediate", num_channels**2, num_channels**2, 
                batch_norm=True, activation_type="relu",
            ),

            # output layer
            self.make_layer(
                "output", num_channels**2, num_channels, 
                batch_norm=False, activation_type="tanh",
            ),
        )

    def make_layer(
        self, layer_type, in_channels, out_channels, 
        batch_norm=True, activation_type="relu",
    ):
        
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
        return self._model(x)
    

class Model_Discriminator(torch.nn.Module):
    """
    convolutional super-resolution model

    """
    def __init__(self, num_channels=3):

        super().__init__()

        self._model = torch.nn.Sequential(

            # input layer
            self.make_layer(
                "input", num_channels, num_channels**2, 
                max_pooling=False, activation_type="relu",
            ),

            # downsampling layer: 240 x 256 -> 60 x 64
            self.make_layer(
                "downsample", num_channels**2, num_channels**2, 
                max_pooling=True, activation_type="relu",
            ),

            # downsampling layer: 60 x 64 -> 15 x 16
            self.make_layer(
                "downsample", num_channels**2, num_channels,
                max_pooling=True, activation_type="relu",
            ),

            # output layers for classification
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=num_channels*15*16, out_features=1),
            torch.nn.Sigmoid(),
        )

    def make_layer(
        self, layer_type, in_channels, out_channels, 
        max_pooling=True, activation_type="relu",
    ):
        
        if layer_type == "downsample":
            layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1,
            )
        else:
            layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            )

        activation = torch.nn.ReLU() if activation_type == "relu" else torch.nn.Tanh()

        return torch.nn.Sequential(
            layer, torch.nn.MaxPool2d(kernel_size=2), activation,
        ) if max_pooling else torch.nn.Sequential(
            layer, activation,
        )

    def forward(self, x):
        return self._model(x)
    