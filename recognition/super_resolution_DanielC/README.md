# Pytorch implmenetation of a brain MRI super-resolution network using ADNI dataset
## 1. General Overview of Super-resolution Network
The Super-resolution network is designed to take in a 2d low resolution brain MRI image as input and return a higher resolution of the image.
The core component of this network is the efficient sub-pixel layer, which is defined within pytorch as ``PixelShuffle``. This layer upscales the image by reducing the number of channels of features, in our case, it upscales by a factor of 4.

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

Since ``Image Folder`` can only be used in a directory with a single image folder, ensure all folders are placed as follows:
- data
  - train
      -train_AD
        -AD
      -train_NC
        -NC
  - test
      -test_AD
        -AD
      -test_NC
        -NC

## 3. Data pre-processing
As low resolution of the images are not provided, we create our own low resolution of images and use the given high 240 x 256 as ground truth. 
The images are converted to tensors then we use the resize() function of pytorch transoforms to manipulate the image to the desired size. 
This can be adjusted alongside hyper-parameters in utils.py


## 4. Example inputs & outputs
Given an input of a low resolution MRI image, should return an upscaled version of the image as shown in the figure.
![image of example input and output](https://github.com/DHyunC/PatternAnalysis/blob/topic-recognition/recognition/super_resolution_DanielC/readme_resources/ExampleFigure.PNG)


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

Furthermore, graphing the losses per iterations is shown as below.
![graph of loss per iteration](https://github.com/DHyunC/PatternAnalysis/blob/topic-recognition/recognition/super_resolution_DanielC/readme_resources/lossgraph.PNG)

After approximately 300 iterations, the network does not seem to improve drastically, therefore if a lower training time is desired, the ``num_epochs`` can be reduced from the default of 100.
A value of 100 was chosen for ``num_epochs`` as per the article's recommendation [here](https://keras.io/examples/vision/super_resolution_sub_pixel/)

## 6. Appendix
1. Where to download the dataset: [ADNI dataset for Alzheimer's disease](https://adni.loni.usc.edu/)
2. Guide / Article: [here](https://keras.io/examples/vision/super_resolution_sub_pixel/)
