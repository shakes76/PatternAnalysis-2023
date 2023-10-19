# Improved UNet arcitecture for Image segmentation of the ISIC2018 dataset
## Recognition task
The purpose of the recognition task is to perform image segmentation on the ISIC 2018 dataset. 
For testing on the test set, an average dice coefficent score [^1] of 0.8 is expected.

## Dataset


## Model arcitecture
The model is employed according to [^2] with changes made to acommodate for use on the ISIC2018 dataset. \
This includes changing 3d modules to 2d, and returning the output through a sigmoid function instead of softmax. \
The model uses deep supervision wich helps mitigate the vanishing gradient problem. It does this by integrating
intermediate loss signals into the output. This allows for more efficient flow of gradients during backpropegation. \
As with general UNet arcitecture; skip connections are also used for their ability to improve segmentation quality. \

The network takes a RGB channeled image as input, and outputs a binary segmentation map.

[^2]: https://arxiv.org/abs/1802.10508v1

![model arcitecture](images/model_arcitecture.png)

## example segmentation
As you can see in the picture below. The image segmentation works well in some cases, and not so good in other ca
![example of image segmentation](images/segmentation_example.png)

## preprocessing and training
The image sizes in the dataset varey. This causes problems when loading the data from the pytorch dataloader.
As a work around fo this issue a standard image size can be specified in the macro *IMAGE_SIZE* in ***dataset.py*** .
The path of the dataset should be specified in the following macros in ***dataset.py***:
'''
TRAIN\_DATA\_PATH
TRAIN\_TRUTH\_PATH
TEST\_TRUTH\_PATH
TEST\_TRAIN\_PATH
'''

For training on rangpur[^3] the test dataset had no corresponding groundtruth images. 
As a work around for this The *ISICDataset* in ***dataset.py*** can be specified with a split ratio as well as a boolean train statement. 
This allows the user to specify wich part of the directory he/she wishes 
to use for both training and testing.

example:
'''
train\_data = ISICDataset(img\_dir=TRAIN\_DATA\_PATH, truth\_dir=TRAIN\_TRUTH\_PATH ,split\_ratio=0.9, transform=transform, train=True)
val\_data = ISICDataset(img\_dir=TRAIN\_DATA\_PATH, truth\_dir=TRAIN\_TRUTH\_PATH, split\_ratio=0.9,transform=transform, train=False)
'''

In this example the train data will be allocated the first 90% of the directory. 
And the valuation data will be allocated the last 10%.

## dependencies
|Name   |Version|
|-------|-------|
|Numpy  |1.24.3 |
|Pytorch|2.1.0  |



## References

