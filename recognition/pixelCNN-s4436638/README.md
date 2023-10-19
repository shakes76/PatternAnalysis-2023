# Brain MRI Super-Resolution Network
## Abstract

This project aimed to create an efficient sub-pixel neural network (or ESPCN) as proposed by Shi, 2016, to upscale low resolution images into a higher resolution version. It will use sub-pixel convolution layers to create an array of upscaling filters, which can then be used increase the resolution of our images. In the case of our project, this will be applied to brain MRI scan images - in particular, the ADNI brain dataset will be used to train our CNN. We will take existing images from the dataset, downsample them by 4x, and use these to train our network to upscale them back to the original resolution. This solves the problem of needless computational complexity that comes about from performing these operations in the high resolution space.

## Model Architecture

The used model utilises a combination of convolution layers for feature maps extraction, and sub-pixel convolution layer to collate these maps and upscale our image. Specifically, this ESPCN will use multiple convolution layers, uses the ReLU function for activation, and the PixelShuffle function to aggregate our channels. The diagram below illustrates this model in action.

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/b79ce09f-9464-4734-8f29-090b08ec5295)

According to the diagram, the low-resolution image undergoes a process of generating multiple feature maps, which are subsequently combined into a single channel to produce the high-resolution image. The number of filters created is determined by squaring the upscaling factor, meaning that our project, which upscales by a factor of 4, will result in the creation of 16 filters. Previous research has demonstrated that the Adam optimizer consistently delivers the most favorable outcomes during model training, and the mean square error loss function has proven to be the most effective method for assessing the system's loss. However, our model slightly varies from the example shown above in that we do not perform all the upscaling in one go; instead of upscaling by 4, we do it twice by 2x with our convolutions occuring in between (ie. conv, upscale, conv, upscale). Upon testing this gave enhanced feature extraction, and resulted in increased image quality.

## Dataset and preprocessing

The dataset utilised was the ADNI brain dataset, consisting of brain MRI images sized at 240x256 pixels. As we want to use different images for training and validation, this dataset was split into 2 arrays, with 90% of the image dataset being used for training and the other 10% for validation. These images also had to be downsampled using the Resize function by our upsampling factor of 4, making them 60x64 when being used in the CNN.

## Training and results

The trainer loads in the original ground truth images in batches of 10, at 240x256, to avoid storing the whole dataset in RAM. An example ground truth image:

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/e906d64c-de25-4f03-82ff-7f629b931fa6)

Through the use of the Resize function, these are downsampled by our upsampling factor of 4:

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/5aea520e-f313-4a36-ba7d-471ba140d947)

We then feed this into the model to get our prediction of the reconstructed upsampled image:

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/26954310-bd30-4172-a936-47e245f2a281)

Then at the end of each epoch, we can run this through our optimiser. Other optimisers did not yield as good results as Adam did, so the example by the Keras implementation was followed and Adam utilised with the same learning rate of 0.001. Again, the shown literature showed the most success with the particular loss function of Mean Squared Error (MSE). MSE produced good results, so no other loss functions were tested. 

Examples of each of the ground truth input, downsampled input and reconstructed image output can be observed above at the validation stage (post training), with numerous epochs run. It was observed from these that the model effectively upscaled the low resolution image, particularly in comparison to the low res image - however, there is still notable problems, such as branches in the coral-like structure in the top left being lost. A graph of the performance of both the validation and training sets can be observed below:

![image](https://github.com/CharlieGore/PatternAnalysis-2023/assets/141538622/05827b6f-b1bc-4b6c-8ee9-1e800d8a56f7)

A concern that can arise in models like these is the model overfitting to the training data, which can be observed by  the plateau of the validation loss being significantly higher than that of the training loss, or worse, by arching back up. However, it can be observed that this does not occur in the validation set.

## Dependencies

The following libraries and respective versions have been utilised in this project:

- Numpy : 1.25.0
- Pytorch : 2.0.1
- TorchVision : 0.15.2
- Python : 3.11.0
- Matplotlib : 3.7.1

## Usage

For easy of reproduction of results, accessibility has been added. A parser has been added such that to run the model, one must in a terminal type 'python train.py --data_path "path/to/training/images"' where path/to/training/image is the folder where you have saved the training ADNI image dataset. The same would then by done for predict, using 'python predict.py --data_path "path/to/test/images"', pointing to the test image dataset. Upon running, the ground truths, downscaled images and upsampled reconstructions are saved in the test_images folder which will be created if it does not exist locally. The PSNR  loss, SSIM loss and MSE loss are also saved in .csv files locally.

Usage (predict):
```bash
python predict.py --data_path "path/to/folder/with/test/images/"
```
Usage (train):
```bash
python train.py --data_path "path/to/folder/with/train/images/"
```

The ADNI dataset is structured as follows:
```bash
├──AD_NC
   ├──test
   │  ├──AD
   │  │   └──*.jpeg
   │  └──NC
   │      └──*.jpeg
   └──train
      ├──AD
      │   └──*.jpeg
      └──NC
          └──*.jpeg
```
Where AD and NC folders each contain .jpeg images.










