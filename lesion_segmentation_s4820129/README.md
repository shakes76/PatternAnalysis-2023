## UNET segmentation of ISIC 2018
### Table of Contents
  - [Libraries](#libraries)
  - [Dataset](#dataset)
  - [UNET Architecture](#unet-architecture)
  - [Training procedure](#training-procedure)
  - [Results](#results)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

### Libraries
- Pytorch
- Wandb
- 
### Dataset
The ISIC 2018 dataset used for this project is part of a public data repository. This dataset includes training data in the form of png and jpg images of skin lesions as well as their ground-truth segmentation. The training data has been preprocessed by resizing the images to 256x256, followed by a normalization with the data's respective mean and standard deviation.
Dataset source: (link)
- Training set:
- Test set:
- Validation set:

### UNET Architecture
The UNET architecture is based on the architecture proposed by a paper () published in (). The network was originally used to do 3D image segmentation on the BRATS 2017 dataset (link), but has been restructured in accordance with the task at hand. It consists of four downsampling blocks, one bottom layer and four upsampling blocks:

#### Downsamling block
- Stride convolution: doubles the amount of features in the map and downsamples the image by a factor of two.
- Double convolution: two convolution layers with preactivation, batch normalization and inbetween dropout. Generates feature maps from the downsampled image tensor.

#### Bottom layer
- A downsampling block
- An upsampling block, consisting of an upsampling layer and a convolution that halves the number of features in the produced map.

#### Upsampling block
- Concatenation with corresponding downsampled feature map. 
- Two convolutions, one 3x3 and one 1x1 that halves the number of feature maps

The resulting network has two pathways: a context pathway that extracts increasingly complex features from the dataset, and a localization pathway that localizes where those features can be found within the original images. In the localization pathway, deep supervision is employed to make sure no information from the feature maps from lower layers are lost. If trained for an appropriate number of epochs, this architecture should produce segmented versions of the image training set, which can be used to identify lesions on later patients' skin.

### Training procedure
As proposed in the paper [], a dice similarity loss function is used to train the network. The dice function uses how much the predicted image overlaps with the masked image to quantify how similar the two images are. Pytorch enables training the network on GPUs. To utilize this UQ's Rangpur Cluster (link) was used. However, due to the number of jobs in queue for GPU training all training in this report is done on the test node. 

### Results
 Due to the limitation in training and a batch size of 32 the results here are obtained from the 8 epochs of training. The metrics used to evaluate the model were dice score, average loss per epoch and the accuracy of the model. The accuracy is computed by accepting the predicted image as a 1.0 where the prediction is > 0.5. After 18 epochs, this was how the models prediction compared to the actual mask:

 ![alt text](images/loss46epoch19.png)
