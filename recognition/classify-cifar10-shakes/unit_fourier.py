'''
Unit test for Fourier transforms in PyTorch
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

#parameters
path = '/home/uqscha22/'

# Custom transform
class NormalizeStatistics(torch.nn.Module):
    """
    Normalize the data to sample zero mean, unit variance per channel
    """
    def __init__(self, unit_variance=False):
        super().__init__()
        self.unit_variance = unit_variance

    def __call__(self, x):
        """
        Args:
            x (PIL Image or numpy.ndarray): Image to be transformed.
        Returns:
            Tensor: Transformed image.
        """
        means = torch.mean(x, dim=[-2,-1]) #per channel
        if self.unit_variance:
            stddevs = torch.std(x, dim=[-2,-1]) #per channel
            x = (x - means[:, None, None]) / stddevs[:, None, None]
        else:
            x = (x - means[:, None, None])
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Fourier(torch.nn.Module):
    """
    Discrete Fourier transform of image and center
    Filter applies a filter to the coefficients, assumes center matches shape of filter
    Zero mean sets DC offset coefficient to epsilon (effectively zero)
    as_real_channels returns real and imaginary components as real channels (i.e. 2 * input_channels)
    """
    def __init__(self, center):
        super().__init__()
        self.center = center
        self.zero_mean = True

    def __call__(self, x):
        """
        Args:
            x (PIL Image or numpy.ndarray): Image to be transformed.
        Returns:
            Tensor: Transformed image.
        """
        c, h, w = F.get_dimensions(x)

        x_complex = x + 0j
        Fx = torch.fft.fft2(x_complex, dim=(-2, -1))
        if self.zero_mean:
            Fx[...,0,0] = 1e-12 #remove DC coefficient
        if self.center:
            Fx = torch.fft.fftshift(Fx, dim=(-2, -1)) #move DC to center
        
        return Fx

#--------------
#Data
transform = transforms.Compose([transforms.ToTensor(), 
                                NormalizeStatistics(unit_variance=True),
                                Fourier(center=True),])

trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True) #num_workers=6

#processing
print("Processing data ...")
start_time = time.time()
Fx = 0
Fy = 0
for i, (x, y) in enumerate(train_loader):
    print(x.shape)
    print(x.dtype)
    Fx = x.numpy()
    Fy = y.numpy()
    break
epoch_time = time.time() - start_time
print("Took " + str(epoch_time) + " secs or " + str(epoch_time/60) + " mins in total")

#plot
#image power spects
plt.figure(figsize=(10, 10))

plot_size = 4
for i in range(plot_size**2):
    # define subplot
    plt.subplot(plot_size, plot_size, 1 + i)
    # turn off axis
    plt.axis('off')
    plt.tight_layout() #reduce white space
    # plot raw pixel data
    plt.imshow(np.abs(Fx[i,0,:,:]), cmap='gray')
    plt.title(str(Fy[i]))

plt.show()