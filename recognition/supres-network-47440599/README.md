# Efficient Sub-Pixel CNN

Nathan Leeks: 47440599

## Overview

The purpose of this recognition task is to create a super resolution network model that can upscale a downscaled image of a brain MRI by a factor of 4 and produce a reasonably clear image with minimum loss of detail. The dataset that will be used for the task is the ADNI dataset. The model that will be used is the Efficient Sub-Pixel CNN. 

## Model Architecture

![Untitled](https://github.com/nleek0/PatternAnalysis-2023/assets/109025848/830e7447-ad2b-46a3-9631-47fa5e09766f)


As shown in the image above, the Efficient Sub-Pixel CNN contains 5 layers.

The first layer is an input layer is an input layer, where the image is the input. The next three layers are then the convolution layers, the parameters of which are shown in the image above. The next layer is than the sub pixel convolution layer, which on PyTorch is the Pixel Shuffle function, and this layer is the one that upscales the image. The final layer is then the output produced by the model. 

## Project Structure

Modules.py - Contains the code for the models.

Dataset.py - Contains all the preprocessing needed for the data.

Train.py - Loads the training data set and then trains the model.

Predict.py - Loads the test data set and then displays images that are produced by the model.

Utils.py - Contains all the hyper parameters and all the constants.

## Data Preprocessing

The ADNI Dataset already contains training images and testing images, therefore each folder was used for its respective purpose. The images were also divided by brains that had Alzheimer’s and brains without, however since this purpose of this model is not to act as a classifier, these labels were ignored. 

The images were grey scaled, and a random horizontal flip was added for more variation. It was then downscaled by 4 to be used by the model.  

For the validation set, since the goal of the model is to upscale the image, the original, none down scaled image was used as the validation set to obtain the loss. 

## Training

Using a GTX 1060 for 100 epochs, the time training took approximately 56 minutes. 

![Untitled 1](https://github.com/nleek0/PatternAnalysis-2023/assets/109025848/52a8bc61-0431-4bd5-8e06-89230dc62303)


It can be seen in the graph above that the loss plateaus relatively quickly as few as 5 epochs, the loss drops significantly, then at around 25, it starts to plateau at a loss of around 0.0023. This suggests that this is the limit of the model.

## Reproducibility

The hyper parameters used can be seen in the [utils.py](http://utils.py) file. The training used 100 epochs and had an up-scaling factor of 4. The data loader also used a batch size of 128. When downscaling the images, they were resized to a dimension of 60x64 as this is 4 times less the resolution of the image. 

The optimiser used was the Adam optimised as it has its own scheduler, making it more convenient. A learning rate of 0.001 was used.

The loss function used was the mean squared error loss(MSELoss). The running loss was a result of the cumulative loss for the batch in the data loader.

## Example

![Untitled 2](https://github.com/nleek0/PatternAnalysis-2023/assets/109025848/aeee65f3-80cf-4dde-b6bd-ff2ce1aa7cc3)


In the images above, the top row contains the original image, the middle row contains the down sampled image, and the last row contains the upscaled image by the model. As seen in the images, while there is still a loss in detail from the images produced by the model compared to the original images, it is still reasonably clear.

## Dependencies:

- Python 3.10
- Pytorch 2.0.1
- Torchvision
- Torch.nn(Pytorch’s neural network module)
- Matplotlib(To display Data)
- Numpy(To display Data)

## Improvements

While the images produced by the model are relatively clear, It can be seen that there are some details in the upscaled image that are missing from the original image. It can also be seen in the loss over epoch graph that the loss plateaus relatively quickly, suggesting that this is the limit of the model.

Therefore, in order to improve the results, training a different model would be the best likely improvement to yield better results. 

## Reference List:

[An Overview of ESPCN: An Efficient Sub-pixel Convolutional Neural Network | by zhuo Cen | Medium](https://medium.com/@zhuocen93/an-overview-of-espcn-an-efficient-sub-pixel-convolutional-neural-network-b76d0a6c875e)

[Image Super-Resolution using an Efficient Sub-Pixel CNN (keras.io)](https://keras.io/examples/vision/super_resolution_sub_pixel/)

[ADNI | Alzheimer's Disease Neuroimaging Initiative (usc.edu)](https://adni.loni.usc.edu/)
