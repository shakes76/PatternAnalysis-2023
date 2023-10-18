# Vision Transformer for ADNI
This project explored the use of a Vision Transformer for the classification of Alzheimer's disease in the ADNI brain dataset.

# Contents


# 1. Introduction
The Alzheimer's Disease Neuroimaging Initiative (ADNI) is designed to provide researchers with study data to assist in defining the progression of Alzheimer's disease. The aim of this project was to classify Alzheimer's disease (normal and AD) of the ADNI brain data using a Vision Transformer, based on the architecture presented by the paper [An Image Is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929.pdf) [1].

This project used the dataset that contained 2D slices of MRI data for a patient, with the folder structure:
```
AD_NC
|--train (21,520 total images)
|  |--AD
|  |  |-- 10,400 images
|  |--NC
|  |  |-- 11,120 images
|--test (9,000 total images)
|  |--AD
|  |  |-- 4,460 images
|  |--NC
|  |  |-- 4,540 images
```
This means that there is ~70% of the total data in the training set, with the test set containing ~30%. Note that all images are formatted as ```{Patient ID}_{Slice Number}.jpeg```.

# 2. Vision Transformer Background
## 2.1 ViT Architecture Overview
| Vision Transformer (ViT) | Transformer Encoder |
| :---: | :---: |
| <img src="misc/vit.gif" width="550" height="400"/> | <img src="misc/encoderblock.png" width="175" height="400"/> |

## 2.2 How it works
The Vision Transformer architecture is comprised of a several stages:

1. **Patch and Position Embeddding (Inputs)**:
    - This converts the input image into a sequence of non-overlapping image patches. Each patch is treated as an individual token in the model's input sequence. Position embeddings are added to specify the spatial order of these patches in the image.
2. **Linear projection of flattened patches (Embedded Patches)**:
    - After extracting the patches, this stage projects them into a learnable embedding space. Using embeddings rather than raw pixel values allows the model to capture meaningful  representations of image content.
3. **Norm (Layer Normalisation)**:
    - Normalises the activations of the embedded patches within each layer, ensuring consistent mean and variance. This helps stabilise training and reduces overfitting.
4. **Multi-Head Attention (Multi-Headed Self Attention)**:
    - Enables the model to focus on different parts of the input sequence (patches) and capture complex relationships between patches. It helps the model understand dependencies and context within the image.
5. **MLP (Multi-Layer Perceptron)**:
    - This stage introduces non-linearity into the model and processes the attended features. It consists of linear layers with activations and dropout, allowing the model to perform complex transformations on the attended information.
    - Within the ViT paper [1], the MLP block contains two linear layers with a GELU non-linear activation function between them and a droput layer after each.
6. **Transformer Encoder**:
    - This is a collection of the layers listed above. There are two skip connections inside the encode (the "+" symbols) meaning the layer's inputs are fed directly to immediate layers as well as subsequent layers. The overall ViT architecture is just a number of Transformer encoders stacked on top of each other.
7. **MLP Head**:
    - This is the output layer of the architecture. It converts the learned features of an input to a class output. Since this is a classification problem, this would be called the "classifier head".

## 2.3 Problem It Solves
While the base Transformer architecture has become the standard for natural language processing (NLP) tasks, its applications to computer vision is limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. The paper demonstrates that a Transformer designed for vision (where an image is converted to a sequence of patches then continued as usual) can have comparable or even superior performance on image classification tasks when compared to convolutional networks.

The benefit of using Vision Transformers is that they are not as sensitive to data augmentation as convolutional networks. This means that they can train on smaller datsets. On top of this, Vision Transformers can learn gloabl features of images. This is due to them being able to attend to any part of the image, regardless of its location. This is especially useful for tasks such as object detection.


# 3. Dependencies & Requirements
To run all the files within this repository, a conda environment can be created using the provided
```environment.yml``` file. After installing conda, one can run the following to instantiate a
new environment with the required dependencies:
```
conda env create -n {environment name} -f environment.yml
```
The main packages used in this repository are:
```
Python ver 3.11
PyTorch ver 2.0.1
Matplotlib ver 3.7.1
```

## 3.1 Repository Structure
The structure of the repository is:
```
config.py:          Configure user parameters
driver.py:          Driver script that runs everything else
dataset.py:         Creates datasets and dataloaders
modules.py:         Contains model structure
train.py:           Contains train, validation and test methods
predict.py:         Contains functionality to show usage of model
utils.py:           Contains helper functions
environment.yml:    Contains conda environment
misc:               Contains images for the README
```

## 3.2 How to use the model
To use the model, the ```config.py``` file can be adjusted based on user preferences, then ```driver.py``` can be run either directly in an IDE or through terminal by calling:
```
>>> python driver.py
```
The hyperparameters that a user can change are:
```
# General user preferences
will_save:          True if user wants to save the model
will_load:          True if user wants to load a model
show_model_summary: True if user wants a summary of the model
will_train:         True if user wants model to be trained
will_test:          True if user wants model to be tested

# Dataloader specific parameters
data_path:          Folder where data is stored
batch_size:         Batch size
n_channels:         Number of image channels
image_size:         Desired image size (will be converted to a square shape)
n_classes:          Number of classes in the dataset
data_split:         Ratio between training and validation data

# Change this at user risk
train_mean:         Training data mean (calculated)
train_std:          Training data standard deviation (calculated)
test_mean:          Testing data mean (calculated)
test_std:           Testing data standard deviation (calculated)

# Training specific parameters
n_epochs:           Number of epochs model will be trained for
learning_rate:      Learning rate for model

# Transformer model specific parameters
patch_size:         Patch size of each image patch that model will process
n_heads:            Number of attention heads to use in MSA block
n_layers:           Number of transformer encoder layers
mlp_size:           Dimensionality of MLP
embedding_dim:      Number of hidden dimensions within the MLP
mlp_dropout:        Dropout rate applied within MLP
attn_dropout:       Dropout rate applied to the attention weights in MSA block
embedding_dropout:  Dropout rate applied to token embeddings

# Model storage specific parameters
load_path:          Path to load model from
save_path:          Path to save model to
results_path:       Folder to store images of results
```

# 4. Results
## 4.1 Data Preprocessing
The model created for this project requires square images as input, so that square patches can be created. Each image in the ADNI dataset is 240x256, so the image was initially just resized to 224x224 (this was the size used in the ViT paper).

An example of a patched image using 16x16 patches on the basic preprocessed data can be seen below:

<img src="misc\patched_example.png" width="400" height="400"/>

It is clear to see that for an essentially unprocessed image, many of the patches do not contain useful information.

Thus, further preprocessing needed to be completed.

### 4.1.1 Varying Image Preprocessing
Several data augmentation processes were introduced to improve model accuracy. These are outlined below.
- Normalisation:
    - The mean and std devation of the train and test data were calculated separately and applied to the images during transformation for the dataloaders.
- Random cropping:


## 4.2 Experimentation

### 4.2.1 Basic Model
The model was first trained and tested using the parameters specified by the Vision Transformer paper for the Base version (viT-Base):
```
patch_size = 16         # From paper
n_heads = 12            # Table 1 from [1]
n_layers = 12           # Table 1 from [1]
mlp_size = 3072         # Table 1 from [1]
embedding_dim = 768     # Table 1 from [1]
mlp_dropout = 0.1       # Table 3 from [1]
attn_dropout = 0.0      # Not used in paper
embedding_dropout = 0.1 # Table 3 from [1]
learning_rate = 0.003   # Table 3 from [1]
```

### 4.2.2 Hyperparameter Tuning
The following parameters were kept the same throughout testing:
```
LIST OF HYPERPARAMETERS
```
| Learning Rate | Accuracy |
|-|-|
| 0.01   | X% |
| 0.001  | X% |
| 0.0001 | X% |

| Data Split | Accuracy |
|-|-|
| 0.01   | X% |
| 0.001  | X% |
| 0.0001 | X% |

| Test | Change | Accuracy |
|------|--------|----------|
| Data Split | 0.6, 0.7, 0.9 | X%, X%, X% |
| Learning Rate | 0.1, 0.01, 0.001 | X%, X%, X% |

### Visualisations
The image below is an example plot of accuracies and losses of training and validation data during model training:

<img src="misc\past_results\example_losses_accuracies.png" width="" height=""/>

The image below is an example set of predictions of a trained model on a random sample of test images:

<img src="misc\past_results\example_predictions.png" width="600" height="400"/>

## 4.3 Reproducibility of Results
- Due to the dynamic nature of the training and validation data split (the )
- discussion of overall effectiveness as well?

# 5. Future Direction
For future training and testing, it is recommended that the PyTorch tool 

# 6. References
https://arxiv.org/pdf/2010.11929.pdf

https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten

https://arxiv.org/pdf/2106.10270.pdf

https://arxiv.org/pdf/2210.07240.pdf

https://www.v7labs.com/blog/vision-transformer-guide#h2

https://www.learnpytorch.io/05_pytorch_going_modular/#4-creating-train_step-and-test_step-functions-and-train-to-combine-them

https://wandb.ai/dtamkus/posts/reports/5-Tips-for-Creating-Lightweight-Vision-Transformers--Vmlldzo0MjQyMzg0
