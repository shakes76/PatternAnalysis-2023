# Improved UNet arcitecture for Image segmentation of the ISIC2018 dataset
## Algorithm
The model is employed according to [^1] with changes made to acommodate for use on the ISIC2018 dataset. \
This includes changing 3d modules to 2d, and returning the output through a sigmoid function instead of softmax. \
The model uses deep supervision wich helps mitigate the vanishing gradient problem. It does this by integrating
intermediate loss signals into the output. This allows for more efficient flow of gradients during backpropegation. \
As with general UNet arcitecture, skip connections are also used for their ability to improve segmentation quality. \

The network takes a tensor of shape (batch\_size, 3, width, height) as input, and returns  
a tensor with shape (batch\_size, 1 width, heigh)
the algorithm takes a batch of 2d RGB channeled images as an input, and outputs a one channeled binary probability 
as an output. 

This project allows for binary image segmentation of rgb channeled mole images
## Inner working

## example segmentation
![example of image segmentation](segmentation_example.png)

## dependencies
|Name   |Version|
|-------|-------|
|Numpy  |1.24.3 |
|Pytorch|2.1.0  |


[^1]: https://arxiv.org/abs/1802.10508v1
