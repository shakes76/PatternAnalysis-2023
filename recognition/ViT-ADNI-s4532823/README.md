# Alzheimer's Disease Classification using a Vision Transformer (ViT)
A Vision Transformer (ViT) was used to classify Alzheimer's disease from MRI brain scans. The [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu) brain dataset was used, and test accuracy of 67% was achieved. This fell short of the 80% target set out in the COMP3710 Report Spec Sheet.

## Model Architecture
The Vision Transformer from Dosovitskiy's paper "An Image is Worth 16x16 Words" was used [1]. The figure below illustrates the model architecture.

A custom implementation of the PyTorch Vision Transformer Base Patch 16 model is used, with no pretrained model weights. The implementation is strongly based off of the PyTorch vision transformer modules [2]. And the MLP module used within the EncoderBlock is from a live coding YouTube video which was "very much a clone" [3] of the Hugging Face vision transformer implementation [4].

![Figure 1](/recognition/ViT-ADNI-s4532823/assets/Model_architecture_Dosovitskiy.png)

The ViT takes a 224x224 image as an input, then splits it into 16 patches. Each patch is linearly embedded using a convolution, then positional embeddings are added, as well as classification tokens [1]. The embedded patches are then passed to a standard Transformer encoder [1]. The Transformer encoder consists of alternating layers of Multi-Head Attention and MLP blocks [1]. A Layernorm is applied before every block, and residual connections after every block [1]. As stated above, the classification head is replaced with a fully connected layer with two class outputs (0 or cognitive normal, 1 for Alzheimer's disease).

## Data Loading & Preprocessing
### About the Dataset
The ADNI dataset was obtained from the COMP3710 Blackboard site, as well as from the Rangpur HPC.

The training set has 21,520 images total. 11,120 cognitive normal (NC) brain scans and 10,400 Alzheimer's disease (AD) scans.

The test set has 9,000 images total. 4,540 cognitive normal (NC) brain scans, and 4,460 Alzheimer's disease (AD) scans.

The dataset contains 20 scans for each patient. And obviously for a given patient all 20 images will be in the same directory (e.g. ./train/AD). 


### Data Loading
The ADNI dataset is used in training and testing of the model dataset is loaded from different paths based on different contexts. 
+ On Linux (i.e. the Rangpur HPC), it's loaded from `/home/groups/comp3710/ADNI/AD_NC`
+ On Windows (i.e. my PC), it's loaded from `D:\ADNI\AD_NC`

The paths above can be altered as necessary in the `dataset.py` file.

A custom data loader was created to load the dataset, with the ability to split the training set into a training and validation set, with an 80% training and 20% validation split. 

As implied in the previous subsection, images are loaded by patient ID (prefix of each file name) to ensure all images from any given patient are in the same set. This prevents data leakage from the training set to the validation set. Images in the testing set are already separate from the training set (by patient ID)

Additionally, one can randomly remove a certain percentage of the data from the data loader. This was used to speed up code execution when creating and debugging train.py. When this percentage is unspecified, none of the data is removed.


### Preprocessing
The training images are preprocessed using the following data transforms (to combat overfitting and ensure data compatibility witht the model):
+ A `RandomHorizontalFlip`
+ A `RandomCrop` to a 224x224 image
+ Conversion to a tensor

The validation and training images are preprocessed to a lesser extent (to ensure data compatibility):
+ A `CenterCrop` to a 224x224 image
+ Conversion to a tensor

## Training & Results
### Training
The Vision Transformer model, as described above, is used in training. 

The model is trained for 20 epochs, with early stopping implemented if a chosen metric (i.e. validation accuracy) decreases three times in a row. 

Below are all hyperparameters of the model

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 20 (or until early stopping) |
| Criterion      | `CrossEntropyLoss` |
| Optimiser      | `Adam` |
| Initial learning rate | 1e-4 |
| Learning rate scheduler  | `LRStep` |
| Scheduler step | 0.5 every five epochs (LR halved every five epochs) |
| Encoder layers | 12 |
| Patch size | 16 |
| Classification outputs | 2 |

Shown below are loss & accuracy plots from a run of `train.py`. It's clear to see that the model is neither overfitting or underfitting to the training set, however further fine tuning is possible to train the model. For example, the use of pretrained model weights, such as the `IMAGENET1K_V1` weights, would improve performance.

![](/recognition/ViT-ADNI-s4532823/assets/loss_plot.png)

![](/recognition/ViT-ADNI-s4532823/assets/accuracy_plot.png)

During training, a status update at the end of epoch is also printed to the terminal.

### Testing & Results
Testing can be carried out in either `train.py` or `predict.py`. 

#### Using `train.py`
Training occurs on the ADNI dataset, then the model trained on the final epoch is used. The model attains 66.56% accuracy across the whole test dataset.

#### Using `predict.py`
The best performing model parameters from training are saved to `adni_vit.pt`, which can then be used to load an instance of the `VisionTransformer` model. A certain number (by default 9) of images are randomly sampled from the test dataset. 

Then each image is labelled with the predicted classification (AD or NC) and actual classification. All displayed in a generated figure, as below

![](/recognition/ViT-ADNI-s4532823/assets/predictions.png)

## Dependencies & Reproducability
### Dependencies
On Windows, Python 3.11.4 and PyTorch 2.0.1 within a Miniconda base environment are used. 

On Linux (i.e. Rangpur HPC), Python 3.11.4 and PyTorch 2.0.1 are again used, but instead within a Miniconda virtual environment.

In addition, the packages `torchvision`, `pillow`, `numpy` and `matplotlib` are used within this implementation. All dependent packages can be installed using the following:
```
pip install pytorch torchvision pillow numpy matplotlib
```

CUDA may also need to be installed on the system of use, to ensure this implementation can be run on 

### Running the Code
+ First change the working directory to `recognition/ViT-ADNI-s4532823`
+ Run `python train.py` in the terminal to train and test the model.
+ Run `python predict.py` in the terminal to run predictions on random images from the test dataset.

Ensure you are using the correct interpreter or have activated the correct virtual environment before running. 

## References

[1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani,
M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale,” arXiv:2010.11929 [cs], Oct. 2020, arXiv: 2010.11929. [Online].
Available: http://arxiv.org/abs/2010.11929

[2] PyTorch contributors, "torchvision."
Available: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

[3] mildlyovefitted, "Vision Transformer in Python." 
Available: https://www.youtube.com/watch?v=ovB0ddFtzzA

[4] Wrightman, R. et. al. (Hugging Face contributors), "pytorch-image-models"
Available: https://github.com/huggingface/pytorch-image-models