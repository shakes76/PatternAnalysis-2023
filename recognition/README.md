# Classify Alzheimer’s disease of the ADNI brain data using perceiver transformer
### Rohan Kollambalath

## Introduction

In this project I use the perceiver transformer to classify if MRI brain scans have Alzheimers disease with the ADNI dataset. The goal of the project was to achieve above 80% accuracy in the test set and produce a model that does not make too many assumtions about the input data.

The Perciever Transformer is an iteration of the transformer model released by Google Deep-Mind to solve its quadratic bottleneck and modalility limitations. [Perceiver: General Perception with Iterative Attention] (https://arxiv.org/pdf/2103.03206.pdf)

it solves the quadratic bottleneck by projecting the input information into a lower dimention latent array through cross attention layers similar to those in traditional transformers. The array is then processed through multiple self attention layers to extract features. This process also allows the perciever to make less structural assumptions about the input data allowing for many different modes of data as inputs including text, audio, video etc. 

![PercieverTransformer](./plots/perciever_transformer.jpg)


## Set-Up and Dependencies
This project uses pytorch to build the perceiver in the same regard as the original paper. I used object oriented programming for all the components of the model and dataloader. 
Architectural decisions for this model were influenced by "The Annotated Perciever" article [2]: [The Annotated Perceiver](https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb) as well as the original paper [1]: [Perceiver: General perception with iterative attention. In International conference on machine learning](https://arxiv.org/abs/2103.03206)

Dependencies:
 - Python 3.9.2
 - PyTorch 2.0.1
 - Matplotlib 3.7.1

## Implementation

### Data Processing

### Perceiver Modules

### Image Encoding
Due to the nature of the perciever it is permuation invarient and cannot directly draw on spatial structures as well as other models. To mitigate this the paper reccomended two options, fourier positional encodings or a learned array of embeddings. This project opts to use the learned embeddings.


### Classification

## Results

### Appendix
#### A: Pre-Processed Data from Dataset
#### B: Hyperparameters 

## References
<a id="1">[1]</a>
Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., & Carreira, J. (2021, July). Perceiver: General perception with iterative attention. In International conference on machine learning (pp. 4651-4664). PMLR.
<a id="2">[2]</a>
Curt Tigges. (Aug 20, 2022). "The Annotated Perceiver." Medium. [Link](https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb)
