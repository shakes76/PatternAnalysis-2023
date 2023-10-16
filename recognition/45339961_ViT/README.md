# Classifying Alzheimer's Disease of the ADNI brain data using a Visual Transformer

# Contents

## Data Background


## Vision Transformer Background
### ViT Architecture Overview
| Vision Transformer (ViT) | Transformer Encoder |
| :---: | :---: |
| <img src="misc/vit.gif" width="550" height="400"/> | <img src="misc/encoderblock.png" width="175" height="400"/> |

The Vision Transformer architecture is comprised of a several stages:

1. **Patch and Position Embeddding (Inputs)**:
    - This converts the input image into a sequence of non-overlapping image patches. Each patch
    is treated as an individual token in the model's input sequence. Position embeddings
    are added to specify the spatial order of these patches in the image.
2. **Linear projection of flattened patches (Embedded Patches)**:
    - After extracting the patches, this stage projects them into a learnable embedding space.
     Using embeddings rather than raw pixel values allows the model to capture meaningful 
     representations of image content.
3. **Norm (Layer Normalisation)**:
    - Normalises the activations of the embedded patches within each layer, ensuring consistent mean
    and variance. This helps, stabilises training and reduces overfitting.
4. **Multi-Head Attention (Multi-Headed Self Attention)**:
    - Enables the model to focus on different parts of the input sequence (patches) and capture 
    complex relationships between patches. It helps the model understand dependencies and context 
    within the image.
5. **MLP (Multi-Layer Perceptron)**:
    - This stage introduces non-linearity into the model and processes the attended features. 
    It consists of linear layers with activations and dropout, allowing the model to perform complex 
    transformations on the attended information.
    - Within the ViT paper [1], the MLP block contains two linear layers with a GELU non-linear
    activation function between them and a droput layer after each.
6. **Transformer Encoder**:
    - This is a collection of the layers listed above. There are two skip connections inside the 
    encode (the "+" symbols) meaning the layer's inputs are fed directly to immediate layers
    as well as subsequent layers. The overall ViT architecture is just a number of Transformer
    encoders stacked on top of each other.
7. **MLP Head**:
    - This is the output layer of the architecture. It converts the learned features of an input to 
    a class output. Since this is a classification problem, this would be called the "classifier head".

### How it works

### Problem it solves


## Dependencies & Requirements

## Results

## References
https://arxiv.org/pdf/2010.11929.pdf

https://www.akshaymakes.com/blogs/vision-transformer

