import torch
import torch.nn as nn



class UNET(nn.Module):
    """UNet model for image segmentation."""
    def __init__(self, n_classes):
        """Initializing the channels, and the workflow for encoding and decoding."""
        super(UNET, self).__init__()

        self.n_classes = n_classes

        def encode_block(in_channels,out_channels):
            path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
            return path
        
        def decode_block(in_channels,out_channels):
            path = nn.Sequential(
                nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(2*out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),  
            )
            return path

        # The encoding path 
        self.encode1 = encode_block(3,64)
        self.encode2 = encode_block(64,128)
        self.encode3 = encode_block(128,256)
        self.encode4 = encode_block(256,512)
        self.encode5 = encode_block(512,1024)

        # The decoding path
        self.decode1 = decode_block(1024,512)
        self.decode2 = decode_block(512,256)
        self.decode3 = decode_block(256,128)
        self.decode4 = decode_block(128,64)
        self.decode_output = decode_block(64, self.n_classes)

            
    def forward(self, x):
        """Passes the data through the encode and decode paths.

        Parameters: 
            x (tensor): the data as a tensor.

        Returns: 
            output (tensor): the output tensor image.

        """
        encode1 = self.encode1(x)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        encode5 = self.encode5(encode4)

        decode1 = self.decode1(encode5)
        decode1 = torch.cat((decode1,encode4),dim=1)
        decode2 = self.decode2(decode1)
        decode2 = torch.cat((decode2, encode3),dim=1)
        decode3 = self.decode3(decode2)
        decode3 = torch.cat((decode3,encode2),dim=1)
        decode4 = self.decode4(decode3)
        decode4 = torch.cat((decode4, encode1),dim=1)
        output = self.decode_output(decode4)

        return torch.sigmoid(output)
    

        

        

        

        




