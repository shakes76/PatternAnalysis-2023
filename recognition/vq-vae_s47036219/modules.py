import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate_channels=None):
        super(ResidualBlock, self).__init__()

        if not intermediate_channels:
            intermediate_channels = in_channels // 2

        self._residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._residual_block(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings  # Save as an instance variable
        self.embedding = nn.Embedding(self.num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)

        # Now x_flat is [batch_size * height * width, channels]
        
        # Calculate distances
        distances = ((x_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2).sum(-1)
        
        # Find the closest embeddings
        _, indices = distances.min(1)
        encodings = torch.zeros_like(distances).scatter_(1, indices.unsqueeze(1), 1)
        
        # Quantize the input image
        quantized = self.embedding(indices)
        
        # Reshape the quantized tensor to the same shape as the input
        quantized = quantized.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        return quantized

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()

        self.encoder = Encoder()
        self.conv1 = nn.Conv2d(64, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.conv1(enc)
        quantized = self.vector_quantizer(enc)
        
        dec = self.decoder(quantized)
        return dec
    
    
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    
    sigma1_sq = (img1 - mu1).pow(2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = (img2 - mu2).pow(2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1)*(img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    
    ssim_n = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    ssim_d = (mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_val = ssim_n / ssim_d

    return ssim_val.mean()