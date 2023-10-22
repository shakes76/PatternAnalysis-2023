# Segmentation of the ISIC 2018 Dataset with the Improved UNet

## Author
Name: Mitchell Keefe

Student ID: 45574539

Project Task: 1

## Dataset

### Description
The International Skin Imaging Collaboration (ISIC) datasets contain dermoscopic images, with challenges being set each year in hope to improve melanoma diagnosis. For this project, the ISIC 2018 challenge dataset will be used, containing 2594 images of skin lesions.

### Pre-processing
The training, validation, test split used in this project was a 70%, 15%, 15% ratio respectively - 1816 training images, 389 testing images, and 389 validation images. This was chosen because it is a widely accepted data split ratio. It is suitable because the ISIC 2018 dataset is not small (> 2000 images). Therefore, allowing for good usage of the available images. During the pre-processing stage, images were resized to (256px, 256px) to keep them uniform. The dimensions were selected as they were close to equalling some of the smallest images in the set. This allowed for training time to be enhanced while, also, maintaining the quality of the images. Images were then normalized, and saved into tensorflow datasets to be used in the training process.

## Improved UNet Architecture

### Description
The UNet is a convolutional network which is used for image segmentation, with its name given because of the U-shaped architecture. It is comprised of encoding and decoding sections. The improved UNet architecture [1] ___. It makes use of convolution layers with a stride of 2 between the context modules to allow for more features while encoding. The localization pathway takes feature from deeper levels in the network which have been encoded in lower resolution, and decode them in higher spatial resolution. For this process, it upsamples / upscales, then has a 3x3x3 convolution which halves amount of feature maps. The upsampled features are then recombined with the results from the element-wise sums performed on the context modules on the same level via concatenation. It also makes use of softmax activation for the outputs rather than sigmoid which is used generally in the traditional UNet architecture.


### Dice Similarity Coefficient
>The Sørensen–Dice coefficient is a statistic used to gauge the similarity of two samples. [2]

For the purpose of this project, the target is for all labels to have a minimum Dice similarity coefficient of 0.8 on the test set.

Note: Reference [3] was used as a resource when implementing the function in this project (information used is under the heading 'Defining custom metrics and loss functions').

## Results
----

## Usage
To replicate results, create a folder named DATA in the project folder. Download the ISIC 2018 Segmentation Task Data: this can be accessed via the COMP3710 Blackboard page -> Course Help/Resources -> ISIC 2018 Segmentation Task Data. Once the 3 folders have been successfully downloaded and stored into the DATA folder created earlier, the necessary dependencies listed under the Dependencies heading below must be installed. For more information on these and the versions required, please refer to the section below. Once the dependencies have been installed, the model can be trained by running the main function in train.py.

## Dependencies
Python 3.9

Tensorflow 2.14.0

Keras 2.14.0

Numpy 1.23.1

Matplotlib 3.8.0

Scikit-Learn 1.3.1

Glob

## References
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online]. Available: https://arxiv.org/abs/1802.10508v1

[2] Wikipedia contributors. "Sørensen–Dice coefficient." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 29 Aug. 2023. [Web]. Available: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

[3] Notebook Community contributors. "Image Segmentation with tf.keras", Notebook Community, 2020, [Web]. Available: https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
