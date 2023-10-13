# Pytorch implmenetation of a brain MRI super-resolution network using ADNI dataset

## 1. Required Dependencies
Dependency | Version |
--- | --- 
python | 3.10
pytorch | 2.1.0
numpy | 1.25.2
matplotlib | 3.5.3
torchaudio | 2.1.0
torchvision | 0.16.0

For reproducibility, a virtual environment is recommended with specified versions respective to dependencies for isolation. 

## 2. Example inputs & outputs
Given an input of a low resolution MRI image, should return an upscaled version of the image as shown in the figure.
![image of example input and output]


## 3. Training & minimizing loss
The padding for convolution network was calculated through: input size - kernel size + 2* padding size / 1 (stride is always 1)
(source https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks)

Changing the input and output channel size of conv2d layers in the model can help to decerase the loss.
However, after a certain point the difference is negligible as shown.

Channel size | Loss 
--- | --- 
32 | 0.00253
64 | 0.00249
128 | 0.00236
