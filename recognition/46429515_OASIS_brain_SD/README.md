#OASIS brain dataset - Stable Diffusion Task


##Stable Diffusion
Stable Diffusion is a type of diffusion model that works in a two step process

that uses a technique similar to how a generative model would work. The diffusion

model operates by gradually adding noise to an input image in a forward process

and then retrieves the original image by denoising (backwards process). In the

(parametrized) backwards process, the model predicts the noise added in each of image 

and generates new datapoints using a neural network. This diffusion model

will be using a U-Net for the backwards process


##Dependencies


##Usage Example
### Dataset Creation
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

##Justification


