# Brain MRI super-resolution network on the ADNI brain dataset
Recognition task: Project 5 - Image super resolution

## Description
This implementation is a Sub-Pixel CNN-based approach for single image super-resolution. The goal is to enhance the resolution of images, making them clearer and more detailed. 

The chosen algorithm for this is the ESPCN (Efficient Sub-Pixel CNN), found at [this link](https://keras.io/examples/vision/super_resolution_sub_pixel/), which achieves the upscaling effect using sub-pixel convolution.

## How It Works
The image of the model architecutre is from [this](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf) paper on ESPCN.

![model architecture from paper](https://github.com/mhjos/PatternAnalysis-2023/blob/topic-recognition/recognition/%20Super-Resolution_46804158/Figures/model.PNG)

The input is down-sampled by a factor of 4 and then using PyTorch's 'PixelShuffle' it increases the resolution, achieving a 4x upscaling effect. This should produce a "clearer" image. 
The model learns the transformation through a series of convolutional layers. The loss fucntion calculates the mean squared error (implemented in PyTorch as torch.nn.MSELoss) between the model's output and the target input. The loss is then used to perform backpropagation and optimize the model. The Adam optimizer algorithm adjusts the learning rate for each model parameter during training.

To use this solution:
- Run train.py to train the ESPCN model using the ADNI brain dataset.
- Run predict.py to apply the trained model to the test dataset and visualize the super-resolved images.

## Dependencies

- Python Python 3.11.4 
- Matplotlib 3.7.1
- Pytorch 2.0.1
- Pytorch Torchvision 0.15.2

A GPU is recommended for faster training, as it significantly reduces training time.

## Reproducibility
To reproduce the results, you'll need to download the ADNI brain dataset externally and specify the dataset paths in the dataset.py file. The trained model can be saved and loaded to apply the super-resolution model to new images.
- [ADNI MRI Dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)


## Pre-processing
The input training, and test data images are centered, resized, and normalized to make them consistent and easy to compare. For more details on the data preprocessing, refer to the dataset.py file.


## Example Inputs, Outputs, and Plots
The provided visualizations include an image of example input, target, and output for the super-resolution model, along with a graph illustrating the model's loss during training.

![image of example input, target and output](https://github.com/mhjos/PatternAnalysis-2023/blob/topic-recognition/recognition/%20Super-Resolution_46804158/Figures/Image.png)

![graph of loss](https://github.com/mhjos/PatternAnalysis-2023/blob/topic-recognition/recognition/%20Super-Resolution_46804158/Figures/Loss.png)

## Data Split
About 30% of the data is reserved for testing, which is a reasonable portion for evaluation while the majority of the data (70%) to be used for training the model. This split is often used in machine learning because it provides a fair way to evaluate your model without using too much data for testing, which can be wasteful.

Training Data (Total: 21,522 images):
- Train AD: 10,401 images (approximately 48.3% of the training data)
- Train NC: 11,121 images (approximately 51.7% of the training data)

Testing Data (Total: 9,002 images):
- Test AD: 4,461 images (approximately 49.5% of the testing data)
- Test NC: 4,541 images (approximately 50.5% of the testing data)
