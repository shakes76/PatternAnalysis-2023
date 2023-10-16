# COMP3710 Project Siamese Network on ADNI(46684033)
## Project aim
Create a classifier based on Siamese network to classify either Alzheimer’s disease (normal and AD)
of the ADNI brain data set, while having an accuracy of around 0.8 on the test set

## Siamese Network
A Siamese neural network is a neural network that uses the
same weights while working in tandem on two different
input vectors to compute comparable output vectors[1].
![img.png](img.png)
*Example for a siamese network*[2]

Siamese neural network has different applications, one of the prevalent usage is matching the inputs to a vector space.
This enables metric learning and can be used to identify inputs' similarity. In this project, the inputs are brain images either with 
Alzheimer’s disease (AD) or Cognitive Normal (CN) from the __ADNI__ dataset. Based on the similarity output of the 
Siamese neural network, brain images can be classified into AD or CN based on similarity on other images with known classes.
## Methodology
### Dataset
The dataset used in this project is the **ADNI** dataset. The Alzheimer's Disease Neuroimaging Initiative (ADNI) is a large and long-term research project that began in 2004 with the primary goal of studying Alzheimer's disease (AD) and related neurodegenerative disorders. 
The dataset structure is as follows:
- ADNI_AD_NC_2D
  - AD_NC
    - test
      - AD
        - *.png
      - NC
        - *.png
    - train
      - AD
        - *.png
      - NC
        - *.png

As the Siamese neural network requires two inputs, `dataset.py` contain a data loader function which return 
two different images, and labeling them 1 if the classes are the same, 0 if the classes are different. For example,
((AD,AD),1) , ((AD,NC),0), ((NC,AD),0), etc.

Images are pre-processed by a number of pytorch transform function to facilitate training process. Specific
transform used is discussed in the discussion section.

### Siamese neural network architecture
The model is defined in `modules.py`. The model consists of a several sequences of convolutional layers, batch normalization layers,
ReLU activation function, and pooling layers. Then followed by several fully connected layer with ReLU activation function in between.

### Training
In `train.py`, the model is trained with the training dataset prepared in `dataset.py`. 

Contrastive loss is used for the training:
![img_1.png](img_1.png)
It takes as input a pair of sample that are either similar or dissimilar, and it brings similar samples closer and
dissimilar samples far apart.

Adam optimizer is used for the training

![training_loss.png](..%2F..%2F..%2F..%2F..%2F..%2FDownloads%2Ftraining_loss.png)
The loss reduces over each epoch, indicating the success and evidence of the training.

### Prediction
After training, the model can be used in predicting the class of an image. The performance of the model is evaluated in
`predict.py`, where images from the test dataset is inputted into the model. To make a good prediction,

## Result
The best result obtained over numerous trial is 78.7% test accuracy on test set. 
## Discussion


## Reference
[1] Chicco, Davide (2020), "Siamese neural networks: an overview", Artificial Neural Networks, Methods in Molecular Biology, vol. 2190 (3rd ed.), New York City, New York, USA: Springer Protocols, Humana Press, pp. 73–94, doi:10.1007/978-1-0716-0826-5_3, ISBN 978-1-0716-0826-5, PMID 32804361, S2CID 221144012

[2] Rao, S.J., Wang, Y., & Cottrell, G. (2016). A Deep Siamese Neural Network Learns the Human-Perceived Similarity Structure of Facial Expressions Without Explicit Categories. Cognitive Science.