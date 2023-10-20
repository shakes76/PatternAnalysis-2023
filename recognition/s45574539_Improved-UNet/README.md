# Segmentation of the ISIC 2018 Dataset with the Improved UNet

## Author
Name: Mitchell Keefe

Student ID: 45574539

Project Task: 1

## Dataset

### Description
The International Skin Imaging Collaboration (ISIC) datasets contain dermoscopic images, with challenges being set each year in hope to improve melanoma diagnosis. For this project, the ISIC 2018 challenge dataset will be used, containing 2594 images of skin lesions.

### Pre-processing

### Training, Validation, Test Split
The training, validation, test split used in this project is a 70%, 15%, 15% ratio respectively. This was chosen because it is a widely accepted data split ratio, and is suitable because the ISIC 2018 dataset is not small (> 2000 images). This split allows for good usage of the available images.

## Improved UNet Architecture

### Description
The UNet is a convolutional network which is used for image segmentation, with its name given because of the U-shaped architecture. It is comprised of encoding and decoding sections. The improved UNet architecture [1] ___.

### Dice Similarity Coefficient
>The Sørensen–Dice coefficient is a statistic used to gauge the similarity of two samples. [2]

For the purpose of this project, the target is for all labels to have a minimum Dice
similarity coefficient of 0.8 on the test set.

## Usage
----

## Results
----

## Dependencies
Python 3.9

Tensorflow 2.14.0

Keras 2.14.0

Numpy 1.23.1

Matplotlib 3.8.0

## References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1

[2] Wikipedia contributors. "Sørensen–Dice coefficient." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 29 Aug. 2023. [Web]. Available: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient


