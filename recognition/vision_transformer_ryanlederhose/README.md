# Classifying Alzheimer's Disease Using a Visual Transformer

The aim of this project is to classify Alzheimer's disease of the ADNI brain dataset using a vision transformer (ViT). The ADNI brain dataset is composed of two classes - AD meaning Alzheimer's disease, and NC meaning Normal Cognitive. The goal is to perform classification with a minimum accuracy on the test dataset of 0.8.

## Vision Transformer
Vision Transformers, often referred to as ViTs, signify a groundbreaking departure from the traditional Convolutional Neural Networks (CNNs) in the realm of computer vision. These models are primarily characterized by their utilization of the Transformer architecture, originally designed for natural language processing but adapted for visual data processing.

![alt text](images/Vision_Transformer.gif)

In a ViT the first step involves dividing an input image into non-overlapping patches of fixed size, typically 16 by 16 pixels. These patches are then linearly embedded into low-dimensional vectors treated as a sequence of tokens. After these patches are linearly embedded a class token is prepended to the sequence. The class token is a learnable embedding focused on representing the entire image, and is hence crucial in classification. All embedded tokens are then added with a positional embedding which is generally just a random tensor representing the location of the embedding on the image.

This sequence of embedded vectors is then processed through a transformer encoder. Within the encoder are multiple feed-forward layers and self-attention layers. This self-attention mechanism is crucial as it allows the model to learn the long-range depedencies between patches. The output of the encoder is fed into a MLP which classifies the image.

## ADNI Dataset
It was mentioned there were two classes in the ADNI dataset, Alzheimer's disease and Normal Cognitive. Below there are examples for both classes (AD top, NC bottom)

![alt text](images/AD_Example.jpeg)

![alt text](images/NC_Example.jpeg)


## Pre-processing
The ADNI dataset provided was given in training and test folders. So, one of the first pre-processing steps used was to seperate the train folder into a training and validation set. It was decided a 80/20 train/validation split would be applied to the data. This is a common split ratio so it was a clear choice to use. On top of this, a patient-level split is also applied. As there are multiple scans of the same patient in the dataset, it was important to ensure that a patient scans only lies in the training or validation set. This is because the model would learn the patients number and hence, if the same patient existed in both datasets, the validation loss (and accuracy) would become a meaningless performance metric. 

When loading images into Python using PyTorch's ImageFolder class, the images undergo several transformations. These being:
 * Image resizing to 240px x 240px 
 * Normalisation with mean 0.5, and standard deviation 0.5
 * Image crop about the centre

The images were resized and cropped about the centre for a size of 240px x 240px because the original size of the image did not faciliate a patch size of 16px x 16px, so it was appropriate to resize the images. Furthermore, the images were normalised as this is a standard pre-processing technique. The values for the mean and standard deviation of normalisation were altered; however, it did not appear to have any significant impact on the results. A batch of brain scans after pre-processing are shown below:

![alt text](images/brains-after-processing.png)

## Model Implementation
The designed model follows the following architecture:

![alt text](images/report-flowchart.drawio.png)

The architecture above describes the general flow of a vision transformer; however, to segment the images into patches, a convolutional layer is used. It is believed this convolutional layer improves performance. As such, the hyperparameters of the model are:

* Patch size --> length and width of image patches in pixels
* Number of encoders --> number of encoder layers in the transformer encoder section
* Number of attention heads --> number of heads on the multi-head attention block
* Hidden size --> number of output channels from the convolutional layers
* MLP Dimension --> dimension of the MLP in the transformer encoder
* MLP Head Dimension --> dimension of the MLP head
* Dropout regularisation
* Number of epochs
* Learning rate
* Batch size
* Optimiser
* Loss Criterion
* Learning rate scheduler
* Weight decay 

## Training and Hyperparameter Tuning
The hyperparameters of an initial training run were:
 * Patch size = 16
 * Number of encoders = 6
 * Number of attention heads = 4
 * Dropout regularisation = 0.1
 * Number of epochs = 60
 * Learning rate = 1e-3
 * Batch size = 32
 * Hidden size = 64
 * Optimiser = Adam
 * Loss Criterion = Cross entropy loss
 * Learning rate scheduler = ReduceLROnPlateau
 * Weight decay = 0.0
 * MLP Dimension = 768
 * MLP Head Dimension = 768

The plot below shows the cross entropy loss of the training at each epoch:

![alt text](images/losses1.png)

From this, it was clear that the model was overfitting to the training data. To try and rectify this, more dropout layers were implemented, and the dropout rate was increased to 0.2. Also, the learning rate was decreased to 1e-4, and the number of encoders was increased to 10. The model responded as below:

![alt text](images/losses2.png)

![alt text](images/accuracy2.png)

The model is still overfitting but the changes have decreased the losses in the validation dataset. To try and introduce more regularisation, weight decay was introduced at 1e-4. Additionally, the number of neurons in the feed-forward layers of the transformer encoder was increased, so that the MLP dimension is 1024.

![alt text](images/losses4.png)
![alt text](images/accuracy4.png)

The model is still overfitting. At this point, I decided to take a different approach. As there is significantly less data in the dataset than a usual deep learning model is trained on, I decided to make the model more simple. All regularisation (dropout and weight decay) was removed, the number of encoder layers was decreased to 4, the image patch size was decreased to 4, the MLP dimension was decreased to 128, and the MLP head dimension was decreased to 512. These hyperparameters produced the following results:

![alt text](images/losses5.png)
![alt text](images/accuracy5.png)

The plots show the model is not overfitting to the same extent (this is evident by the lowered training accuracy and increased cross entropy loss). This trained model performed admirably on the test dataset, with an accuracy of 69.65%.

![alt text](images/test_accuracy5.png)

## Final Model Design and Test Accuracy

```python
'''
HyperParameters

This class stores all the hyperparameters for the vision transformer
'''
class HyperParameters(object):
    def __init__(self) -> None:
        self.patch_size = 8             # image patch size
        self.mlp_dim = 128              # dimension of mlp in transformer encoder
        self.head_dim = 1024            # dimension of mlp head
        self.n_channels = 3             # number of channels for convultional layer
        self.num_encoders = 4           # number of transformer encoders
        self.num_heads = 4              # number of attention heads
        self.dropout = 0.0              # dropout regularisation
        self.num_classes = 2            # number of classes
        self.epochs = 60                # max number of epochs
        self.lr = 1e-3                  # learning rate
        self.weight_decay = 0.00        # weight decay regularisation
        self.batch_size = 32            # batch size
        self.hidden_size = 64           # size of convolutional layer
```

## Example Use of Model
```predict.py``` shows an example use of the trained model by taking a random batch of the test dataset, showing this batch, and then predicting the classes to test accuracy.