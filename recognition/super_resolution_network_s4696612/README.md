# Brain MRI Super-Resolution CNN
## COMP3710 Report - Pattern Recognition - Jarrod Mann - s4696612

## Introduction
The project aimed to create a deep learning model that could sufficiently upsample a low
resolution image. Specifically, the project focused on upsampling brain MRI scans. Creating an
effective model for this task would mean less overall storage space would be required for the
scans while they were not actively being used. Instead, a low resolution image could be stored,
and then be processed through the model each time its use was required. Therefore, the model
aims to reconstruct brain MRI scan images to as high a detail as possible.

## Model Implementation
An efficent sub-pixel convolutional neural network was implemented to complete the project.
This model consists of multiple normal convolutional layers, (activated with the rectified
linear unit function), and a pixelshuffle operation. In this model, the convolutions are
applied to the low resolution image before any upsampling is performed. The convolutions
generate a number of channels equal to the square of the upscaling factor; 16 filters are
made. The pixelshuffle opperation then 'shuffles' the components of these channels into 1
channel, thus creating a high resolution image. Through training, the convolutional layers
learn to supply the pixelshuffle with result channels that accurately represent the original 
image, and the pixelshuffle operation learns the way the data should be arranged to
successfully recreated the high resolution image. In this way, super-resolution is achieved.

The Adam optimiser was used in model training. The SGD optimiser was also tested, however,
it quickly became apparent that the Adam optimiser achieved better resulting images. Similarly,
mean squared error loss was used to measure the model's accuracy during training, but absolute
error was also tested. After early testing, it was concluded that mean squared error was more
effective for training the model, thus it was used in the final implementation.

## Dataset
The ADNI brain dataset was used for the training, validation and testing of the model. The
ADNI image size is 240x256. These images were each downsampled by a factor of 4 to create
datasets with 'input' images of size 60x64; the original image was used as the 'label' for
model development.

## Training
The loss for the training data and validation set throughout training were recorded and compared.
The model's loss quickly converged to approximately 0.09 in all trials. This can be seen in the
below plot of the losses. In this plot, the orange line indicates the validation set's loss, and
the blue line is the training set's loss. The model is not overly overfitted since the validation
set's loss is very similar to the training set. Furthermore, the test set had similar loss of 
approximately 0.1, indicating that the model successfully upscales the MRI scans.

![Plot of training/validation losses](/recognition/super_resolution_network_s4696612/training_loss.png)

## Results
The trained model achieves its goal of significantly increasing the resolution of brain MRI scans.
However, it is not a flawless model. A comparison of the input, model output, and goal images can be
seen below. This sample was chosen randomly, but it readily illustrates the model's effectiveness.
It should be noted that while the images have been scaled, they same visible dimensions below; no change has
been made to their pixels, which results in evident differences in image quality.

![Comparison of input, output and goak](/recognition/super_resolution_network_s4696612/Comparison.png)

Evidently, the model greatly improves image resolution - the model generated image is far closer to
the goal image than the input. It was observed that the generated model illustrates many of the details
evident in the goal. However, it is evident that this model does not portray the intensity of greys
very well - the black background is slightly grey. This issue suggests further improvements to the
model should be investigated.

Further samples of model generated images and their goal images are shown below. The goals are in
the same position as the model images in the second grid.

![Sample model output](/recognition/super_resolution_network_s4696612/model_sample.png)
![Sample model goal](/recognition/super_resolution_network_s4696612/goal_images.png)


## Dependencies
The project implements deep learning models using the following libaries. Functionalities
may not be supported for other library versions.
| Library  | Version |
| ------------- | ------------- |
| Python | 3.11.4 |
| Pytorch | 2.0.1 |
| PIL | 9.4.0 |
| Torchvision | 0.15.2 |
| Numpy | 1.25.2 |
| Matplotlib | 3.7.2 |

Additionally, the project code is dependent on the location of the dataset within the machine
running the code. To run the files in this project, the paths in utils.py must be changed to 
appropriately direct to the dataset. The path to the "train" portion of the dataset should be
set as "train_path", and the "test" portion should be "test_path". Since the dataset was not
provided with a validation set, a validation set was constructed within train.py as a random
10% of the training data.