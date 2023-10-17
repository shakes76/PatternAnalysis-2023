# Brain MRI super-resolution network on the ADNI brain dataset
Recognition task: Project 5 - Image super resolution

## Description
This implementation is a Sub-Pixel CNN-based approach for single image super-resolution. The goal is enhancing the resolution of images, making them clearer and more detailed. 

The algorithim Efficient Sub-Pixel CNN (https://keras.io/examples/vision/super_resolution_sub_pixel/) which uses sub-pixel convolution.

## How It Works
The input is down-sampled by a factor of 4 and then uding PyTorch as PixelShuffle' it increases the resolution, achieving a 4x upscaling effect. This should produce a "clearer" image. The loss fucntion calculates the mean squared error between the model's output and the target input. The loss is then used to perform backpropagation and optimize the model.

To use this solution:
- Run train.py to train the ESPCN model using the ADNI brain dataset.
- Run predict.py to apply the trained model to the test dataset and visualize the super-resolved images.

## Dependencies

- Python 3.7 or higher
- Matplotlib 3.3.4
- Pytorch
- Pytorch Torchvision

A GPU is recommended for faster training, as it significantly reduces training time.

## Reproducibility
To reproduce the results, you'll need to download the ADNI brain dataset externally and specify the dataset paths in the dataset.py file. The trained model can be saved and loaded to apply the super-resolution technique to new images.
- ADNI MRI Dataset: https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI


## Pre-processing
The training, and test data images are centered, resized, and normalized to make them consistent and easy to compare. For more details on the data preprocessing, refer to the dataset.py file.


## Example Inputs, Outputs, and Plots
This model was run on a mac laptop, and is therefore not very efficient as an improved model would take too long to run. I recognize that the images prodcues are not very good, however with a GPU you could increase the accuracy of the images with simple alterations to the model. 

![image of example input, target and output] (https://github.com/mhjos/PatternAnalysis-2023/blob/topic-recognition/recognition/%20Super-Resolution_46804158/Figures/Figure_1.png)

![graph of loss] ()

## Data Split
About 30% of the data is reserved for testing, which is a reasonable portion for evaluation while the majority of the data (70%) to be used for training the model. This split is often used in machine learning because it provides a fair way to evaluate your model without using too much data for testing, which can be wasteful.

Training Data (Total: 21,522 images):
- Train AD: 10,401 images (approximately 48.3% of the training data)
- Train NC: 11,121 images (approximately 51.7% of the training data)

Testing Data (Total: 9,002 images):
- Test AD: 4,461 images (approximately 49.5% of the testing data)
- Test NC: 4,541 images (approximately 50.5% of the testing data)
