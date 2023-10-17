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

## 2.2 How it works
Here is an example patched image:

<img src="misc\patched_example.png" width="600" height="600"/>

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

# 4. Results

## 4.1 Data Preprocessing
- needs to include train/validation split.

## 4.2 Experimentation

### 4.2.1 Basic Model
- basic model from paper, using hyperparameters specified from paper.

### 4.2.2 Fine-Tuning model
- explanation of changing loss function / learning rate and model shape

## 4.3 Reproducibility of Results
- discussion of overall effectiveness as well?

# 5. References
https://arxiv.org/pdf/2010.11929.pdf

https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten

https://arxiv.org/pdf/2106.10270.pdf

https://arxiv.org/pdf/2210.07240.pdf

https://www.v7labs.com/blog/vision-transformer-guide#h2

https://www.learnpytorch.io/05_pytorch_going_modular/#4-creating-train_step-and-test_step-functions-and-train-to-combine-them

https://wandb.ai/dtamkus/posts/reports/5-Tips-for-Creating-Lightweight-Vision-Transformers--Vmlldzo0MjQyMzg0
