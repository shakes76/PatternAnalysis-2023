# UNet Segmention on the ISIC 2017/8 dataset

### Description: 
The task is to preform image segmentaion on the ISIC 2017/8 dataset. The implementation will employ a UNet convolutional network model for accurate segmentation. The UNet will take the given images and their coresponing Ground truth masks to which it will try to predict a mask. 

### Usage 

The UNet model can segment any type of image if it has enough traing data for both images and the ground truth masks.Therfore the model can be used on any dataset not just this one. In this senario we are using the UNet to segment skin lesions. This is benefical for medical imaging, containing the lesion to then take it further and detect irregularities in the lesion.

### ISIC 2017/8 dataset
The dataset consits of 2000 lesion images and 2000 ground truth masks. Preview of the first 10 images and their respective ground truth masks below.

 <img width="435" alt="first photos" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/47b7a4ef-abae-4622-a5a9-0ab5065ca35d"> 

<img width="440" alt="first masks" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/28c63790-50c7-40c7-94fd-443d288bfbb1">

### Architecture

![u-net-architecture](https://github.com/mraula/PatternAnalysis-2023/assets/96328895/14b488d2-e7bd-477e-a1b9-846d7e157e10)

#### Encoder Block

The encoder block has two 3x3 convolution layers with a ReLU then a max pooling. The encoder block is used to store features aswell as lower the dimentions of the image. The copy and crop is used as skip features in the decoder block from the stored features.

#### Bridging Block

The bridging block applies the two  two 3x3 convolution layers with a ReLU.

#### Decoder Block 

The decoder block has two 3x3 convolution layers with a ReLU then a up conv 2x2. The 4 encoder outputs are used as skip features in each decoder block. The Decoder tires to recunstruct based on the imput and skip features given.