# Medical Image Segmentation using UNet

## Introduction

The conducted project attempted the following: Segment the ISIC 2018 data set with the Improved UNet with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

This project aims to perform medical image segmentation using the UNet architecture. Image segmentation is a crucial task in medical imaging, where the goal is to identify and outline regions of interest in images, such as tumors, blood vessels, or organs. In this project, we use the UNet architecture, which is known for its effectiveness in biomedical image segmentation tasks. In this case the model was used to analyse data from the ISIC 2018 dataset returning the regions of interest on images of skin lesions. The ISIC 2018 dataset can be found at https://challenge.isic-archive.com/data/#2018.

## Model Architecture

The core of our project is the UNet architecture, which consists of an encoder and a decoder. The encoder captures the essential features of the input images, while the decoder generates pixel-wise segmentation masks. The architecture includes convolutional layers, batch normalization, dropout, and max-pooling for down-sampling, as well as transposed convolutional layers for up-sampling. The model architecture is implemented in the `modules.py` file.

## Data Preprocessing

I used the ISIC 2018 dataset. This dataset can be swapped out with any other custom dataset when running the `dataset.py` file, to load and preprocess the training data. The dataset includes image and mask pairs, where masks represent ground-truth segmentations. Data preprocessing involves resizing images and normalizing masks to facilitate model training. The data images must be stored as .jpgs while the masks must have the same name as the corresponding image file except with _segmentation following the name and it must be stored as a png.

## Training

The training process is defined in the `train.py` file. Key components of the training process include loss computation (Dice loss), backpropagation, and optimization using the Adam optimizer. Training is executed over multiple epochs, with periodic validation to track the model's performance. This code runs the validation and tests on the same training data by splitting the data up, 70% for testing, 15% for validation, and 15% for testing. This was done because some work was done on the 2018 ISIC dataset as well which doesn't have a separate validation set. This can be changed with some minor tweaking of the code, but for the sake of using this data, you only need to supply 1 set of data for training, validation, and testing.

### Training Evidence

- Batch Size: 4
- Learning Rate: 0.0001
- Number of Epochs: 11

## Results

The project's primary output is the segmentation of medical images, which is saved in the `predictions` directory. These predictions can be further evaluated and utilized for various medical applications. The training results produced:

Training and Validation results: 

Epoch 1/10 - 1835s 4s/step - loss: 0.4112 - dice_coef: 0.3725 - val_loss: 0.3663 - val_dice_coef: 0.3975

Epoch 2/10 - 1854s 4s/step - loss: 0.3495 - dice_coef: 0.4902 - val_loss: 0.2826 - val_dice_coef: 0.5648

Epoch 3/10 - 1809s 4s/step - loss: 0.2666 - dice_coef: 0.6294 - val_loss: 0.2219 - val_dice_coef: 0.6705

Epoch 4/10 - 1833s 4s/step - loss: 0.2257 - dice_coef: 0.6868 - val_loss: 0.2304 - val_dice_coef: 0.7107

Epoch 5/10 - 1898s 4s/step - loss: 0.1891 - dice_coef: 0.7366 - val_loss: 0.2077 - val_dice_coef: 0.7663

Epoch 6/10 - 2053s 5s/step - loss: 0.1755 - dice_coef: 0.7567 - val_loss: 0.2010 - val_dice_coef: 0.7650

Epoch 7/10 - 2867s 6s/step - loss: 0.1610 - dice_coef: 0.7741 - val_loss: 0.1861 - val_dice_coef: 0.7921

Epoch 8/10 - 3356s 7s/step - loss: 0.1578 - dice_coef: 0.7785 - val_loss: 0.1870 - val_dice_coef: 0.7903

Epoch 9/10 - 4022s 9s/step - loss: 0.1512 - dice_coef: 0.7876 - val_loss: 0.1644 - val_dice_coef: 0.7932

Epoch 10/10 - 4703s 10s/step - loss: 0.1442 - dice_coef: 0.7968 - val_loss: 0.1673 - val_dice_coef: 0.7999




Testing results:

Mean Dice Coefficient: 75.47

## Discussion

The UNet architecture is a powerful tool for medical image segmentation. Its ability to capture fine details in images makes it particularly suitable for identifying subtle features in medical images. However, the success of the model depends on the quality and quantity of the training data.

When I started this project, I started it in another repository. For all the commit logs see this repository at https://github.com/SamWolfenden/Lab-Report which has been made public, or see the screenshots in this repository.

## Test Driver Script

To run the model:
1. Save the ISIC 2018 dataset into the downloaded PatternAnalysis-2023 folder.
2. Run 'python train.py'
   - This will train the model and save the model to the desired location.
3. Run 'python predict.py'
   - This will run the testing images through the model and print the result of the model.

## Requirements/Dependencies

- Python
- numpy
- tensorflow
- cv2
- glob
- sklearn


## References

Image segmentation. (2023). Image segmentation. TensorFlow. https://www.tensorflow.org/tutorials/images/segmentation

ISIC Challenge. (2016). Isic-Archive.com. (https://challenge.isic-archive.com/data/#2017)

Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, Klaus H. (2017). Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. ArXiv.org. (https://arxiv.org/abs/1802.10508v1)


