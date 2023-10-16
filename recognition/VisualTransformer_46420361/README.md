# Using a Vision Transformer to Classify Alzheimer's Disease

## ADNI Brain MRI Dataset
The ADNI Brain dataset consists of a training and testing folder, with each subfolder AD and NC containing a collection of patients that are either positive (AD/Alzheimers group) or negative (NC/Healthy group).

![Alt text](images/data_samples.png)

## Vision Transformers
Transformer models are models which uses an 'attention mechanism' to 'remember' previous parts of a sentence, to be able to create complex and coherent sentences and paragraphs. Vision transformers take the same concept but for images instead. This is done by creating embedding vectors of sub-parts of an image, known as a patch. The model then uses an attention mechanism and positional encodings to know where in the image each part belongs.

![Alt text](images/vit_diagram.png)

## Architecture
### Patch Embedding Layer

### Multi Head Self Attention Block

### Machine Learning Perceptron Block

### Transformer Block


## What problem does it solve?
This model attempts to classify Alzheimer's disease from the ADNI mri brain dataset
## Results
### Data Splitting

### Reproducing results
```
conda env create -n <environment name> -f environment.yml
```

## References
https://www.akshaymakes.com/blogs/vision-transformer?fbclid=IwAR2Wmo7_nlLg2EILO6vsKAYucaTl-AXK7NSGY_LBYTP3aPvU_uWW7iF4dVc

https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

https://www.google.com/url?sa=i&url=https%3A%2F%2Fpaperswithcode.com%2Fmethod%2Fvision-transformer&psig=AOvVaw1a8kPLGrK_UEKC5jbPwaJC&ust=1697543404952000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCOibv47A-oEDFQAAAAAdAAAAABAE