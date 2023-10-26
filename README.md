## COMP_3710_Report

## Medical condition, Extended to 27 OCT

## This report is focused on the first task (a)
   Segment the ISIC data set with the Improved UNet
   with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

## The structure of Improved UNet is based on the paper. 
   "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge"
   https://arxiv.org/abs/1802.10508v1

   U-Net is a convolutional neural network architecture primarily used for biomedical image segmentation. 
   Its U-shaped structure consists of a contracting path, which captures context, and an expansive path,
   which enables precise localization. Through skip connections, features from the contracting path are concatenated
   with the expansive path, enhancing localization capabilities.

## Dataset
   In this report, the ISIC 2018 dataset will be used. 
   The ISIC 2018 dataset is a publicly available dataset for skin lesion image segmentation,
   provided by the International Skin Imaging Collaboration (ISIC). Given that the real-world
   images in the dataset come in different sizes, they are uniformly resized to a 128x128 dimension.
   These images use RGB with 3 color channels for input. The label data, which indicates where the lesions are,
   is treated in the same way as the real data. However, these labels are input as grayscale images with a single channel,
   making them simpler and more focused on the lesion's location and shape.

## Model
   






