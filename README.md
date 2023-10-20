# Medical Image Segmentation using UNet

## Introduction

The conducted project attempted the following: Segment the ISIC 2017/8 data set with the Improved UNet with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

This project aims to perform medical image segmentation using the UNet architecture. Image segmentation is a crucial task in medical imaging, where the goal is to identify and outline regions of interest in images, such as tumors, blood vessels, or organs. In this project, we use the UNet architecture, which is known for its effectiveness in biomedical image segmentation tasks. In this case the model was used to analyse data from the ISIC 2017 dataset returning the regions of interest on images of skin lesions.

## Model Architecture

The core of our project is the UNet architecture, which consists of an encoder and a decoder. The encoder captures the essential features of the input images, while the decoder generates pixel-wise segmentation masks. The architecture includes convolutional layers, batch normalization, and max-pooling for down-sampling, as well as transposed convolutional layers for up-sampling. The model architecture is implemented in the `modules.py` file.

## Data Preprocessing

I used the ISIC 2017 dataset. This dataset can be swapped out with any other custom dataset when running the `dataset.py` file, to load and preprocess the training data. The dataset includes image and mask pairs, where masks represent ground-truth segmentations. Data preprocessing involves resizing images and normalizing masks to facilitate model training. The data images must be stored as .jpgs while the masks must have the same name as the corresponding image file except with _segmentation following the name and it must be stored as a png.

## Training

The training process is defined in the `train.py` file. Key components of the training process include loss computation (Dice loss), backpropagation, and optimization using the Adam optimizer. Training is executed over multiple epochs, with periodic validation to track the model's performance.

### Training Evidence

- Training Input Directory: `ISIC2018_Task1-2_Training_Input_x2`
- Training Mask Directory: `ISIC2018_Task1_Training_GroundTruth_x2`
- Batch Size: 4
- Learning Rate: 0.1
- Number of Epochs: 10

## Results

The project's primary output is the segmentation of medical images, which is saved in the `predictions` directory. These predictions can be further evaluated and utilized for various medical applications. The training results produced:

- Epoch 1/10, Average Dice Coefficient: 0.2782156467437744
- Epoch 2/10, Average Dice Coefficient: 0.19667883217334747
- Epoch 3/10, Average Dice Coefficient: 0.16954968869686127
- Epoch 4/10, Average Dice Coefficient: 0.15880490839481354
- Epoch 5/10, Average Dice Coefficient: 0.14603781700134277
- Epoch 6/10, Average Dice Coefficient: 0.14193236827850342
- Epoch 7/10, Average Dice Coefficient: 0.1289714127779007
- Epoch 8/10, Average Dice Coefficient: 0.1291133463382721
- Epoch 9/10, Average Dice Coefficient: 0.12223245203495026
- Epoch 10/10, Average Dice Coefficient: 0.12053724378347397

Images produced by `predict.py` are grey images, with no black and or white and therefore the `predict.py` code does not work as expected.

## Discussion

The UNet architecture is a powerful tool for medical image segmentation. Its ability to capture fine details in images makes it particularly suitable for identifying subtle features in medical images. However, the success of the model depends on the quality and quantity of the training data.

## Test Driver Script

To run the model:
1. Run 'python train.py'
   - This will train the model and save the model to the desired location.
2. Run 'python predict.py'
   - This will run the testing images through the model and output the product to the desired location.

## Requirements/Dependencies

- Python
- PyTorch
- torchvision
- PIL (Pillow)

- Directories must be inserted into `train.py` and `predict.py` files.

## References

ISIC Challenge. (2016). Isic-Archive.com. (https://challenge.isic-archive.com/data/#2017)

Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, Klaus H. (2017). Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. ArXiv.org. (https://arxiv.org/abs/1802.10508v1)
