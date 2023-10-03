# OASIS brain dataset - Stable Diffusion Task


## Stable Diffusion

Stable Diffusion is a type of diffusion model that works in a two step process that uses a technique similar to how a generative model would work. The diffusion model operates by gradually adding noise to an input image in a forward process and then retrieves the original image by denoising (backwards process). 

In the (parametrized) backwards process, the model predicts the noise added in each of image  and generates new datapoints using a neural network. This diffusion model will be using a U-Net for the backwards process


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

With a custom dataset class created (OASISDataset), this will enable the images to be transformed as desired, as well as implement the dataset into a dataloader to be used for our task, where our specified root_path is the path to the parent folder of images of the dataset and the batch size is 32.

```
train_data = OASISDataset(root=f'{root_path}/keras_png_slices_train', label_path=f'{root_path}/keras_png_slices_seg_train', transform=transform)
test_data = OASISDataset(root=f'{root_path}/keras_png_slices_test', label_path=f'{root_path}/keras_png_slices_seg_test', transform=transform)
validate_data = OASISDataset(root=f'{root_path}/keras_png_slices_validate', label_path=f'{root_path}/keras_png_slices_seg_validate', transform=transform)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
validate_loader = DataLoader(validate_data, batch_size=batch_size)
```

### Noise Scheduler (Forward Process) - module.py

In this process, we build more gradually noisy images to be inputted into our model. Here, noise-levels/varianes are pre-computed and we sample each timestep image separately (Sums of Gaussians = Gaussian). The output of the noisy images can be seen as follows (code referenced from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL):

```
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
    
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

image = next(iter(train_loader))[0]

plt.figure(figsize=(15, 15))
plt.axis('off')
num_images = 10
step = int(T/num_images)

for idx in range(0, T, step):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/step) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)
```


### U-Net (Backwards Process) - module.py

In the backwards process, we create a U-Net model to predict the nose in the image where the input is a noisy image (coming from forward process) and the output is the noise in the image. For the U-Net, it is a simple network which consists of a downsampling, residual sampling (via sinusoidal position embedding) and upsampling of data. To create
a model of the network:

```
model = UNet()
```

The number of parameters in the model can be found as follows (output included):

```
print("Num params: ", sum(p.numel() for p in model.parameters()))
output -> Num params:  21277921
```

The neural network cannot distinguish between each timesteps as the network has its parameters shared across time but it is required to filter out noise of varying intensities. This is worked around by using positional embeddings which stores the noise intensity information. The positions index are calculated using sine and cosine: 

$P(k, 2i) = sin(k/(n^{2i/d}))$, $P(k, 2i + 1) = cos(k/(n^{2i/d}))$

These are added as additional input alongside the noisy image. The following code is used in the U-Net model:

```
self.time_mlp = nn.Sequential(
            PositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
```



### Loss Function - train.py

Diffusion models calculate the distance of the predicted noise and actual noise in the image to determine the loss (denoising score similarity equivalent to variational inference). The loss function is simply used in the training process as follows:

```
loss = get_loss(model, images, t)
loss.backward()
```


### Sampling - train.py

Since noise variances has been pre-calculated for the Noise Scheduler, the noise variances must also be passed into the U-Net for denoising.

In this section, the model is called and subtracts the noise prediction from the current image inputted.

The following is the returned image when denoising, using an equation:

```
model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
```

### Training - train.py

In this section of the code, the Adam Optimizer is used for the training of the model.

```
# Adam Optimizer for training the model
optimizer = Adam(model.parameters(), lr=0.001)
```

When training over epochs, the code goes through two sections:

1. Training Loop
2. Validation Loop

The model goes through training and saves the processed image every 10 epochs, and the model goes through a validation every 5 epochs. In the validation process, if the model is found to be the best so far, the model will be saved for future usage in predict.py

The main section of the training loop relies on the following code:

```
    optimizer.zero_grad()

    t = torch.randint(0, module.T, (batch_size,), device=device).long()
    loss = get_loss(model, batch[0], t)
    loss.backward()
    optimizer.step()
```



## Justification


