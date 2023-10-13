# Pytorch implmenetation of a brain MRI super-resolution network using ADNI dataset
## 1. General Overview of Super-resolution Network
The Super-resolution network is designed to take in a 2d low resolution brain MRI image as input and return a higher resolution of the image.
The core component of this network is the efficient sub-pixel layer, which is defined within pytorch as PixelShuffle. This layer upscales the image by reducing the number of channels of features, in our case, it upscales by a factor of 4.

## 2. Required Dependencies
Dependency | Version |
--- | --- 
python | 3.10
pytorch | 2.1.0
numpy | 1.25.2
matplotlib | 3.5.3
torchaudio | 2.1.0
torchvision | 0.16.0

For reproducibility, a virtual environment is recommended with specified versions respective to dependencies for isolation. 

## 3. Example inputs & outputs
Given an input of a low resolution MRI image, should return an upscaled version of the image as shown in the figure.
![image of example input and output](https://github.com/DHyunC/PatternAnalysis/blob/topic-recognition/recognition/super_resolution_DanielC/readme_resources/ExampleFigure.PNG)


## 4. Training & minimizing loss
The padding for convolution network was calculated through: input size - kernel size + 2* padding size / 1 (stride is always 1)
(source https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks)

Changing the input and output channel size of conv2d layers in the model can help to decerase the loss.
However, after a certain point the difference is negligible as shown.

Channel size | Loss 
--- | --- 
32 | 0.00253
64 | 0.00249
128 | 0.00236

Furthermore, graphing the losses per iterations is shown as below.
![graph of loss per iteration](https://github.com/DHyunC/PatternAnalysis/blob/topic-recognition/recognition/super_resolution_DanielC/readme_resources/lossgraph.PNG)
After approximately 175 iterations, the loss no longer decreases consistently rather, it fluctuates around 0.00255 thus we could assume that 175 is the optimal number of iterations. 
