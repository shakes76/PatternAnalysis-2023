# Pytorch implmenetation of a brain MRI super-resolution network using ADNI dataset

## 0. Table of contents
- [Pytorch implmenetation of a brain MRI super-resolution network using ADNI dataset](#pytorch-implmenetation-of-a-brain-mri-super-resolution-network-using-adni-dataset)
  - [0. Table of contents](#0-table-of-contents)
  - [1. General Overview of Super-resolution Network](#1-general-overview-of-super-resolution-network)
    - [1.1 Core functionality](#11-core-functionality)
  - [2. Required Dependencies](#2-required-dependencies)
  - [3. Data pre-processing](#3-data-pre-processing)
  - [4. Example inputs \& outputs](#4-example-inputs--outputs)
  - [5. Training \& minimizing loss](#5-training--minimizing-loss)
  - [6. Appendix](#6-appendix)

## 1. General Overview of Super-resolution Network
The Super-resolution network is designed to present a plausible high-resolution network through the usage of a low resolution input, thus effectively upsampling the given input. Within this specific task, the designed models takes in a 2d low resolution brain MRI image as input and returns a higher resolution of the image. The network is trained on the ADNI brain MRI image dataset, found [here](ADNI dataset for Alzheimer's disease](https://adni.loni.usc.edu/) and aims to increase the resolution by a factor of 4 (from 60x64 to 240 x 256).

### 1.1 Core functionality
The core component of this network is the efficient sub-pixel layer, which is defined within pytorch as ``PixelShuffle``. This layer upscales the image by reducing the number of channels of features, in our case, it upscales by a factor of 4. The network learns the mapping between the low-resolution and their high-resolution version through the various convolution layers by using the mean squared error loss function represented in pytorch as ``torch.nn.MSELoss``. The Adam optimizer is used as per the guide which adjusts learning rate to adapt throughout training. 
The train dataset is split into 75% train set and 25% validation set as validation images are not provided in the ADNI dataset.

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
In utils:
- set ``train_dir`` as the path to your train directory. (Should cotain AD and NC folders)
- set ``test_dir`` as the path to your test directory. (Should cotain AD and NC folders)
- set ``model_path`` as the path you want to save your model to.


## 3. Data pre-processing
As low resolution of the images are not provided, we create our own low resolution of images and use the given high 240 x 256 as ground truth. 
The images are converted to tensors then we use the resize() function of pytorch transoforms to manipulate the image to the desired size. 
This can be adjusted alongside hyper-parameters in utils.py


## 4. Example inputs & outputs
Given an input of a low resolution MRI image, should return an upscaled version of the image as shown in the figure.
![image of example input and output](https://github.com/DHyunC/PatternAnalysis/blob/topic-recognition/recognition/super_resolution_DanielC/readme_resources/newExampleFigure.PNG)


## 5. Training & minimizing loss
The padding for convolution network was calculated through: input size - kernel size + 2* padding size / 1 (stride is always 1)
(source https://stats.stackexchange.com/questions/297678/how-to-calculate-optimal-zero-padding-for-convolutional-neural-networks)

Changing the input and output channel size of conv2d layers in the model can help to decerase the loss.
However, it should be can drastically increase cuda core usage, thus it should be changed from utils if not using a gpu.

Channel size | Loss 
--- | --- 
32 | 0.00253
64 | 0.00249
128 | 0.00236
256 | 0.00223

In addition different activations functions were tested,
Activation Function | Loss 
--- | --- 
Exponential Linear Unit | 0.00261
Sealed Exponential Linear Unit | 0.00264
Leaky ReLU | 0.00247
ReLU | 0.00251

As shown above leaky relu was the best activation function thus it was chosen.

Batch size had a significant effect as lowering it increased accuracy and computational resources which was expected. 
To balance out the two aforementioned factors, a size of 30 is set as default.

Furthermore, graphing the total training loss per epoch is shown as below.
![graph of loss per iteration](https://github.com/DHyunC/PatternAnalysis/blob/topic-recognition/recognition/super_resolution_DanielC/readme_resources/newerExampleFigure.PNG)

As shown in the graph, the rate at which the total loss decreases drastically slows down between 10-15 epochs thus, if a user desired for time effective tranining which yielded acceptable results, stopping at 10-15 epochs would result in similar predictions as 15+ epochs. However to minimize loss and retain practicality, I have chosen to go with a middle ground of 40 epochs.

Afterwards, if the model has performed better than the saved model, it will save the model.


## 6. Appendix
1. Where to download the dataset: [ADNI dataset for Alzheimer's disease](https://adni.loni.usc.edu/)
2. Guide / Article: [here](https://keras.io/examples/vision/super_resolution_sub_pixel/)
