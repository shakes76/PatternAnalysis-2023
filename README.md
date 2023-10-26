## Vision Transformer for the Classification of Alzheimer’s Disease of ADNI Dataset

### Overview/ the Problem

I have implemented a Convolutional Neural Network Vision Transformer (CViT), an adaptation of the standard Vision Transformer (ViT) approach to classify Alzheimer’s disease (normal NC and Alzheimer’s Disease AD) within the ADNI dataset. The notion of a CViT was encouraged by the CvT: Introducing Convolutions to Vision Transformers paper ([link](https://arxiv.org/abs/2103.15808)).

### Background/ Model Overview

Prior to deciding on a CNN-VIT, I thoroughly researched various aspects of transformers to understand their functionality, advantages, and limitations. This led me to the paper ‘Attention Is All You Need’ ([link](https://arxiv.org/abs/1706.03762)), a prevalent paper in the field of deep learning introducing the concept of a transformer. It pains me to admit that this however, was not what drew my eye to this paper, but rather the fact that I shared a name with one of the authors- it’s a rare occurrence for someone named Noam.

This paper introduced a revolutionised way of processing sequences, by solely relying on attention mechanisms, dispensing the need for recurrent layers. As suggested by the title, the utilisation of “self-attention” enables the model to weigh the significance of different parts of an input sequence differently. It is able to capture contextual relations between elements (i.e. pixels), regardless of their position in the sequence. This contextual awareness can lead to more accurate classifications, as the model dynamically adjusts the significance it assigns to various input features based on the information it has learned. In the context of AD classification, the model focuses on crucial parts of the input data that are most indicative of the disease.

#### Vision Transformer (ViT)

A standard ViT breaks down images into fixed-size patches and linearly embeds them as sequences of vectors. This sequence includes an additional ‘class’ token for classification tasks. These sequences are then processed, applying self-attention mechanisms as mentioned above. The output corresponding to the ‘class’ token passes through a feed-forward neural network (FFN) to predict the image’s class. ViT leverages positional embeddings to maintain the image structure information. Whilst this model ushers numerous advantages in image processing, ViTs also yield several limitations in the context of image classification tasks:

1.	Data Efficiency- ViT work best on large, labelled datasets- outperformed by CNNs on smaller datasets
2.	Feature Localisation- ViT treats an image as a sequence of patches, losing explicit local feature representations that are innate to CNNs
3.	Computational Efficiency- self-attention mechanism in ViT computes pairwise interactions between patches (computationally expensive for high-resolution images)
4.	Fine-grained Feature Sensitivity- ViTs may overlook subtle cues due to patch-based processing (relevant in medical image context)- CNNs capture such details more robustly

Integrating CNNs into the model aims to reduce the impact of these limitations. This is achieved by:

1.	Data Efficiency- CViT model can extract hierarchical features better in lower data regimes- leverages inductive biases of CNNs that require less data to generalise well
2.	Localisation of Features- Beginning with convolutional layers, CViT maintains advantages of localised feature extraction- enables transformer part to focus on global feature relationships
3.	Computational Efficiency- CNN used to reduce spatial resolution (and thus sequence length) prior to transformer stage, making attention computations more manageable
4.	Fine-grained Feature Sensitivity- CViT utilise CNN in capturing detailed nuances and global reasoning capabilities of transformers

Merging CNNs and ViTs addresses shortcomings of ViTs by harnessing the strengths of both architectures; it integrates the hierarchal feature learning ability of CNNs with the high-level reasoning capabilities of Transformers. This aims to facilitate the robustness, versatility, and proficiency of the transformer, and hence, outperform traditional ViT or CNN architectures in isolation.

#### Convolutional Vision Transformer (CViT)

My implementation of CViT initiates with a series of convolutional layers, acting as feature extractors; segmenting input images into numerous patches and concurrently learning hierarchical features. This prepares the data for the subsequent transformer architecture, designed to capture complex dependencies and relationships between these patches, irrespective of their spatial positions in the image.

The core of this model, the transformer section, is structured into stages; each of which comprising of one or more transformer blocks. These blocks are integral in handling the model’s reasoning and analytical capabilities, and each block contains multi-head self-attention mechanisms followed by FFNs. This enables the model to focus on different facets of the data and consider various contextual relationships.

Unique to this model is the adaptive configuration of attention heads, MLP ratios, and other hyperparameters across different stages, allowing a more customised approach to learn these hierarchical representations. To ensure the model’s resilience against overfitting and facilitate more stabilised learning, the CViT employs specific regularisation techniques, including layer normalisation and dropout strategies.

The final part of the model compresses the transformer’s output, focusing on a CLS token and passes it through a linear layer that acts as a classifier. This translates the information from preceding stages into concrete predictions for

## ADNI Brain Dataset & Pre-Processing

The project utilises images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset ([ADNI](https://adni.loni.usc.edu)). Each 2D, 256 x 240 pixel image is one of 20 slices from a specific patient’s scan collection.

Data pre-processing is conducted across both `dataset.py` and `process.py`. Both scripts make use of various Pytorch functionalities for streamlined and effective data handling.

#### Dataset.py

This script handles the initial stages of data handling:
- Importing dataset from designated directories
- Constructing a structured DataFrame that organises image file paths alongside their respective labels, providing foundational mapping for subsequent stages

#### Process.py

This script handles more complex data processing, distinguishing its function by:
- Implementing patient-level split by extracting unique patient IDs from image paths; the script ensures a non-overlapping distribution of patients between training and validation sets, preserving the integrity of evaluation
- Conducting data augmentation and normalisation to enhance the robustness of the model
- Facilitating batch processing to expedite the computational process (images batched together during training)

## Training and Validation Performance 

## Usage Description 

## Dependencies

## License

## References