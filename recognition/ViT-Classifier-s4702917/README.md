# Visual Transformer for Classification

This project uses a Vision Transformer model to classify images from the ADNI brain dataset as images of either normal or Alzheimer brains. It makes use of the **TODO** model from the paper **TODO**, and supplied by the **timm** package by Ross Wightman.

## What is a Vision Transformer?

A Vision Transformer is a model that takes in an image, splits it into a number of small (in this case 32x32) chunks, creates a linear array of those chunks, and passes that array of chunks (with each chunck concatenated with a positional embedding vector) as the tokens for a transformer classifier to operate on.

todoo: **TODO: Explanation of Attention**

The reason that Vision Transformers can be more effective than CNNs when given sufficient data comes down to the essential reason that CNNs are effective; they are structured based on the assumption that two pixels next to each other are going to be highly correlated, and that repeatedly analysing local relationships is sufficient to determine the global structure of the image. This is a reasonable bias to introduce into a classifier, since it is broadly true of most images, and allows a CNN to achieve a high amount of accuracy on a small amount of data. ViTs, meanwhile, have no inductive bias towards considering local relationships and to the extent that this is important have to learn it. This means they require more data to get to the same level of accuracy (benefitting significantly more than CNNs from data augmentation), but have the advantage that they can learn exactly the extent to which local relationships are important and no more. Viewing them as statistical constructs, CNNs have a bias introduced to allow them to converge faster, but because they are biased they don't converge towards true predictors of the image labels, whereas ViTs, as unbiased (or at least less biased) estimators converge slower, but do converge towards true predictors of the image labels.

## How does the Vision Transformer perform?

asdfs

## Dependencies

Notable dependencies of this project are as follows:

[//]: # (Check torch version on rangpur.)

|   Package   |   Version    |
| ----------- | ------------ |
| matplotlib  |    3.5.3     |
|    numpy    |    1.23.1    |
|    torch    | 2.0.1+cu117  |
| torchvision | 0.15.2+cu117 |
|    timm     |    0.9.7     |
