# UNet Segmention on the ISIC 2017/8 dataset

### Description: 
The task is to preform image segmentaion on the ISIC 2017/8 dataset. The implementation will employ a UNet convolutional network model for accurate segmentation. The UNet will take the given images and their coresponing Ground truth masks to which it will try to predict a mask. 

### Usage 

The UNet model can segment any type of image if it has enough traing data for both images and the ground truth masks.Therfore the model can be used on any dataset not just this one. In this senario we are using the UNet to segment skin lesions. This is benefical for medical imaging, containing the lesion to then take it further and detect irregularities in the lesion.

### ISIC 2017/8 dataset
The dataset consits of 2000 lesion images and 2000 ground truth masks. Preview of the first 10 images and their respective ground truth masks below.

 <img width="435" alt="first photos" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/47b7a4ef-abae-4622-a5a9-0ab5065ca35d"> 

<img width="440" alt="first masks" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/28c63790-50c7-40c7-94fd-443d288bfbb1">


