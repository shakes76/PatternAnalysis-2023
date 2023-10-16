#  Brain MRI super-resolution network on the ADNI brain dataset
Recognition task: Project 5 - Image super resolution

## Description
Sub-Pixel CNN-based approach for single image super-resolution, enhancing the resolution of images, making them clearer and more detailed. 

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
The images are centered, resized, and normalized. The training, validation, and test data splits are created to ensure robust evaluation. For more details on the data preprocessing, refer to the dataset.py file.


## Example Inputs, Outputs, and Plots
This model was run on a mac laptop, and is therefore not very efficient as an improved model would take too long to run. I recognize that the images prodcues are not very good, however with a GPU you could increase the accuracy of the images with minor alterations. 

### Inputs

- Input images should be in the JPEG format.
- Images should be preprocessed to have a resolution of 240x240 pixels and normalized.

### Outputs

- XYZNet provides probability scores for each disease category.
- A threshold of 0.5 is used to classify images into binary categories (e.g., 'Disease' or 'No Disease').

### Plots






## Data Split
About 30% of the data is reserved for testing, which is a reasonable portion to ensure robust evaluation while allowing the majority of the data (70%) to be used for training the model effectively.

Training Data (Total: 21,522 images):
- Train AD: 10,401 images (approximately 48.3% of the training data)
- Train NC: 11,121 images (approximately 51.7% of the training data)

Testing Data (Total: 9,002 images):
- Test AD: 4,461 images (approximately 49.5% of the testing data)
- Test NC: 4,541 images (approximately 50.5% of the testing data)


