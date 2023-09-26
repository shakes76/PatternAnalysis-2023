# OASIS brain dataset - Stable Diffusion Task


## Stable Diffusion
Stable Diffusion is a type of diffusion model that works in a two step process

that uses a technique similar to how a generative model would work. The diffusion

model operates by gradually adding noise to an input image in a forward process

and then retrieves the original image by denoising (backwards process). In the

(parametrized) backwards process, the model predicts the noise added in each of image 

and generates new datapoints using a neural network. This diffusion model

will be using a U-Net for the backwards process


## Dependencies

* PyTorch: '>=2.0.1'
* Numpy: '>=1.24.3'
* Pillow (PIL): '>=10.0.0'
* Torchvision: '>=0.15.2'
* Matplotlib: '>=3.7.2'

import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image

## Usage Example

### Dataset Creation - dataset.py

With a custom dataset class created (OASISDataset), this will enable the images to be transformed
as desired, as well as implement the dataset into a dataloader to be used for our task, where our
specified root_path is the path to the parent folder of images and the batch size is 32.
```
train_data = OASISDataset(root=f'{root_path}/keras_png_slices_train', transform=transform)
test_data = OASISDataset(root=f'{root_path}/keras_png_slices_test', transform=transform)
validate_data = OASISDataset(root=f'{root_path}/keras_png_slices_validate', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
validate_loader = DataLoader(validate_data, batch_size=batch_size)
```

### Noise Scheduler (Forward Process) - module.py

In this process, we build more gradually noisy images to be inputted into our model. Here,
noise-levels/varianes are pre-computed and we sample each timestep image separately
(Sums of Gaussians = Gaussian)

## Justification


