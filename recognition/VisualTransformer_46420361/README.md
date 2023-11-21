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
This layer uses an attention mechanism to add weights to the input sequence within parallel to the other heads. This allows the model to 'pay attention' to specific features of interest which positively affect the model. The multiple heads allows the model to pay attention to multiple features. The layer norm layer normalizes the patch embeddings across the embedding dimensions. The multi-head attention layer in three learnable vector forms: query, key and value. These values allow each patch to form a relationship with each other patch in the sequence, allowing it to have self attention.

### Machine Learning Perceptron Block
Perceptrons have a similar architecture to that of neurons in the brain. The perceptrons take a multiple weighted inputs and sum them. They then use an activation function to create outputs. For this models particular case, the model uses a fully connected (linear) layer and a gaussian error linear unit (GELU) activation function with a small amount of mlp dropout.

### Transformer Block
The transformer block is simply a combination of the two above blocks (MSA and MLP) and a feed forward mechanism for the blocks.

## What problem does it solve?
This model attempts to classify Alzheimer's disease from the ADNI mri brain dataset. 

More generally speaking, Vision Transformers have many use cases such as object detection, segmentation, image classification and action recognition.

In comparison to convolutional neural networks (CNNs), Vision Transformers are able to outperform them when contextual understanding is crucial in the task at hand and there are large amounts of training data.

## Preprocessing
### Normalization
Finding the mean and standard deviation (std) of the images allows to model to be more effective. The mean and std were calculated as 0.1155 and 0.2224 for training and 0.1167 and 0.2228 for testing, respectively.

### Cropping
As most of the image of the brain scan is black, wasted space, cropping can be added to improve model training. A crop size of 192 was found to be the most effective.

### Resizing
To create patches of an even size, the input image must be square. Therefore, the image must be resized. A size of 256 was found to be the most effective.

### Gray Scale
Since the mri image is grey, pytorch must treat it as if it is such. To do so, the image must be grayscaled so pytorch knows it only has one channel.
## Results
### Data Splitting
The current dataset is split into train and test with AD and NC subfolders. For the purposes of this investigation, a validation dataset is also needed. The base dataset has an approximate split of 70% training and 30% testing. The test data set should not be touched as in a real life scenario we wouldn't have access to this. Therefore, we can split the training dataset, moving 30% of it to validation. The final split is then 50%, 30% and 20% for training, testing and validating respectively.

### Reproducing results
Create environment using conda
```
conda env create -n <environment_name> -f environment.yml
```
Run the main.py file with the same variables and hyperparameters:
```
# data variables
model_name = 'my_model.pth'
root = '/home/callum/AD_NC/'
image_size = 256
crop_size = 192
batch_size = 64
patch_size = image_size // 8
channels = 1
embedding_dims = channels * patch_size**2
patches = (image_size // patch_size)**2
num_heads = embedding_dims // 64
num_classes = 2

# hyperparameters
epochs = 15
learning_rate = 0.001
weight_decay = 0.0001

criterion = CrossEntropyLoss
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```
Run the main.py file
```
cd path/to/repo/recognition/VisualTransformer_46420361/
python main.py
```

### Loss and Accuracy
The loss and accuracy per epoch can be seen in the figure below. It should be noted that the accuracy was still increasing so more improvements could have been made by running more epochs. More epochs weren't run due to computation time.
![Alt text](images/loss_and_accuracy.png)

### Predictions
Using the model, we can make predictions on the test data:
![Alt text](images/predictions.png)

## Dependencies
All the dependencies are stored within the `environment.yml` file, however these are the core dependencies and their respective versions
```
pytorch 2.0.1
numpy 1.25.2
matplotlib 3.7.2
```

## References
### Websites
https://www.akshaymakes.com/blogs/vision-transformer?fbclid=IwAR2Wmo7_nlLg2EILO6vsKAYucaTl-AXK7NSGY_LBYTP3aPvU_uWW7iF4dVc

https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c

https://medium.com/@faheemrustamy/vision-transformers-vs-convolutional-neural-networks-5fe8f9e18efc

https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem

### Images
https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FVision-Transformer-architecture-main-blocks-First-image-is-split-into-fixed-size_fig1_357885173&psig=AOvVaw3naQkwUnuyAVqHPmSygFE2&ust=1697694028496000&source=images&cd=vfe&opi=89978449&ved=0CA4QjRxqFwoTCKiBwKPx_oEDFQAAAAAdAAAAABAD

### Papers
https://arxiv.org/pdf/2010.11929.pdf