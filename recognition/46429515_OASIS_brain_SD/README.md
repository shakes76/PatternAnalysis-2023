# OASIS brain dataset - Stable Diffusion Task

The aim of this model is to generate a brain from a noisy image using the Stable Diffusion Model.

## Stable Diffusion

Stable Diffusion is a type of diffusion model that works in a two step process that uses a technique similar to how a generative model would work. The diffusion model operates by gradually adding noise to an input image in a forward process and then retrieves the original image by denoising (backwards process). 

In the (parametrized) backwards process, the model predicts the noise added in each of the images and generates new datapoints using a neural network. Afterwards, when the noise is done being added, the model will work backwards, predicting the noise added to the image and then subtracting the noise from the image. Eventually, this will result into a clearer, denoised image.

This diffusion model will be using a U-Net for the backwards process.


## Dependencies

* PyTorch: `>=2.0.1`
* Numpy: `>=1.24.3`
* Pillow (PIL): `>=10.0.0`
* Torchvision: `>=0.15.2`
* Matplotlib: `>=3.7.2`


## Usage Example

### Stable Diffusion Generating Outcomes
>> link to images here (predict.py images in a folder into github)
![test screenshot](/recognition/46429515_OASIS_brain_SD/predict_output/generated_image_0.png) ![test screenshot 2](/recognition/46429515_OASIS_brain_SD/predict_output/generated_image_1.png)


### Dataset Creation - dataset.py

With a custom dataset class created (OASISDataset), this will enable the images to be transformed as desired, as well as implement the dataset into a dataloader to be used for our task, where our specified root_path is the path to the parent folder of images of the dataset and the batch size is 32 (can be found in utils.py).

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

The precomputed values can be seen in the module.py section in utils.py
The diffusion noising code can be found in the noise scheduler section in module.py


### U-Net (Backwards Process) - module.py

In the backwards process, we create a U-Net model to predict the nose in the image where the input is a noisy image (coming from forward process) and the output is the noise in the image. For the U-Net, it is a simple network which consists of a downsampling, residual sampling (via sinusoidal position embedding) and upsampling of data. To create a model of the network:

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

The sampling codes can be found in the train.py section of utils.py


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

The Adam optimizer is a popular choice for deep learning modules as it adapts learning rates to each parameters during training, thus selecting a basic learning rate of 0.001 allows for balanced learning between initial progression and fine-tuning together with the optimizer's features.

The batch size was sized to be 32 as the provided OASIS dataset have been sliced into 32 per case. The image size has been selected to be 128x128 as 64x64 is considered to be too small for proper clarity of the image outputs and 256x256 may be too big for some of the images from the dataset provided, thus possibly adding inaccuracy to the model.


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
