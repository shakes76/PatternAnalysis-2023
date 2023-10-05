# OASIS brain dataset - Stable Diffusion Task


## Stable Diffusion

Stable Diffusion is a type of diffusion model that works in a two step process that uses a technique similar to how a generative model would work. The diffusion model operates by gradually adding noise to an input image in a forward process and then retrieves the original image by denoising (backwards process). 

In the (parametrized) backwards process, the model predicts the noise added in each of image  and generates new datapoints using a neural network. This diffusion model will be using a U-Net for the backwards process


## Dependencies

* PyTorch: `>=2.0.1`
* Numpy: `>=1.24.3`
* Pillow (PIL): `>=10.0.0`
* Torchvision: `>=0.15.2`
* Matplotlib: `>=3.7.2`

import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image


## Usage Example

### Dataset Creation - dataset.py

With a custom dataset class created (OASISDataset), this will enable the images to be transformed as desired, as well as implement the dataset into a dataloader to be used for our task, where our specified root_path is the path to the parent folder of images of the dataset and the batch size is 32.

```python
train_data = OASISDataset(root=f'{root_path}/keras_png_slices_train', label_path=f'{root_path}/keras_png_slices_seg_train', transform=transform)
test_data = OASISDataset(root=f'{root_path}/keras_png_slices_test', label_path=f'{root_path}/keras_png_slices_seg_test', transform=transform)
combined_data = ConcatDataset([train_data, test_data])
validate_data = OASISDataset(root=f'{root_path}/keras_png_slices_validate', label_path=f'{root_path}/keras_png_slices_seg_validate', transform=transform)

data_loader = DataLoader(combined_data, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)
validate_loader = DataLoader(validate_data, batch_size=batch_size)
```


### Noise Scheduler (Forward Process) - module.py

In this process, we build more gradually noisy images to be inputted into our model. Here, noise-levels/varianes are pre-computed and we sample each timestep image separately (Sums of Gaussians = Gaussian). The output of the noisy images can be seen as follows (code referenced from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL):

The code can be seen in its section within module.py


### U-Net (Backwards Process) - module.py

In the backwards process, we create a U-Net model to predict the nose in the image where the input is a noisy image (coming from forward process) and the output is the noise in the image. For the U-Net, it is a simple network which consists of a downsampling, residual sampling (via sinusoidal position embedding) and upsampling of data. To create
a model of the network:

```python
model = UNet()
```

The number of parameters in the model can be found as follows (output included):

```python
print("Num params: ", sum(p.numel() for p in model.parameters()))
output -> Num params:  21277921
```

The neural network cannot distinguish between each timesteps as the network has its parameters shared across time but it is required to filter out noise of varying intensities. This is worked around by using positional embeddings which stores the noise intensity information. The positions index are calculated using sine and cosine: 

$P(k, 2i) = sin(k/(n^{2i/d}))$, $P(k, 2i + 1) = cos(k/(n^{2i/d}))$

These are added as additional input alongside the noisy image. The following code is used in the U-Net model:

```python
self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
```


### Loss Function - train.py

Diffusion models calculate the distance of the predicted noise and actual noise in the image to determine the loss (denoising score similarity equivalent to variational inference). The loss function is simply used in the training process as follows:

```python
loss = get_loss(model, images, t)
loss.backward()
```


### Sampling - train.py

Since noise variances has been pre-calculated for the Noise Scheduler, the noise variances must also be passed into the U-Net for denoising.

In this section, the model is called and subtracts the noise prediction from the current image inputted.

The following is the returned image when denoising, using an equation:

```python
# Subtracting noise from image
model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
```


### Training - train.py

In this section of the code, the Adam Optimizer is used for the training of the model.

```python
# Adam Optimizer for training the model
optimizer = Adam(model.parameters(), lr=0.001)
```

When training over epochs, the code goes through two sections:

1. Training Loop
2. Validation Loop

The model goes through training and saves the processed image every 10 epochs, and the model goes through a validation every 5 epochs. In the validation process, if the model is found to be the best so far, the model will be saved for future usage in predict.py.

The respective sections of the trainings can be found inside the train.py file.


## Justification

#justification on why i chose specific values like epochs, learning rates, optimizers (hyper parameters)


## Future Direction

There are multiple ways that this stable diffusion model from scratch can be improved upon. The main methods for significant improvements are:
* Increasing the complexity of the model for the Backwards Process
* Changing from simple UNet to other models such as ResNet or Conditional U-Nets
* Change the beta schedule equation used (sinusoidal, etc.)
* Improve accuracy of the model by applying different transformations to the initial images


## References

### Code referenced from:
* https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL (Heavily Referenced)
* https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb
