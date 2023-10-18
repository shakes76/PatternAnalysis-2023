# Using a Vision Transformer to Classify Alzheimer's Disease

## ADNI Brain MRI Dataset
The ADNI Brain dataset consists of a training and testing folder, with each subfolder AD and NC containing a collection of patients that are either positive (AD/Alzheimers group) or negative (NC/Healthy group).

![Alt text](images/data_samples.png)

## Vision Transformers
Transformer models are models which uses an 'attention mechanism' to 'remember' previous parts of a sentence, to be able to create complex and coherent sentences and paragraphs. Vision transformers take the same concept but for images instead. This is done by creating embedding vectors of sub-parts of an image, known as a patch. The model then uses an attention mechanism and positional encodings to know where in the image each part belongs.

![Alt text](images/vit_diagram.png)

## Architecture
### Patch Embedding Layer
The Patch Embedding Layer involves taking an input image and splitting into equal sized patches. The patches are then converted to a sequence of learnable embedding vectors. A classification token embeding vector is linearly projected with the learnable embedding vector, which later becomes the embeddings label. Finally, a positional embedding is added to the learnable embedding vectors, so the model knows the origin of each patch within the image and the class.

### Multi Head Self Attention Block
This layer uses an attention mechanism to add weights to the input sequence within parallel to the other heads. This allows the model to 'pay attention' to specific features of interest which positively affect the model. The multiple heads allows the model to pay attention to multiple features.

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
### Websites
https://www.akshaymakes.com/blogs/vision-transformer?fbclid=IwAR2Wmo7_nlLg2EILO6vsKAYucaTl-AXK7NSGY_LBYTP3aPvU_uWW7iF4dVc

https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

### Images
https://www.google.com/url?sa=i&url=https%3A%2F%2Fpaperswithcode.com%2Fmethod%2Fvision-transformer&psig=AOvVaw1a8kPLGrK_UEKC5jbPwaJC&ust=1697543404952000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCOibv47A-oEDFQAAAAAdAAAAABAE

### Papers
https://arxiv.org/pdf/2010.11929.pdf