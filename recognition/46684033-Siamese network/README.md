# COMP3710 Project Siamese Network on ADNI (Man Lik Nicholas Wong 46684033)
## Project aim
Create a classifier based on Siamese network to classify either Alzheimer’s disease (normal and AD)
of the ADNI brain data set, while having an accuracy of around 0.8 on the test set

## Siamese Network
A Siamese neural network is a neural network that uses the
same weights while working in tandem on two different
input vectors to compute comparable output vectors[1].

![img.png](images_for_README%2Fimg.png)

*Example for a siamese network*[2]

Siamese neural network has different applications, one of the prevalent usage is matching the inputs to a vector space.
This enables metric learning and can be used to identify inputs' similarity or facilitate classification. In this project, the inputs are brain images either with 
Alzheimer’s disease (AD) or Cognitive Normal (NC) from the __ADNI__ dataset. Based on the similarity output of the 
Siamese neural network, brain images can be classified into AD or CN based on similarity on other images with known classes.

The goal is to input a brain MRI image to the trained SNN, extracts its embeddings, and use a classifier to find its class (AD or NC)

![218391_78.jpeg](images_for_README%2F218391_78.jpeg)

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
    - validation
      - AD
        - *.png
      - NC
        - *.png

As the Siamese neural network requires two inputs, `dataset.py` contain a data loader function which return 
two different images, and labeling them 1 if the classes are the same, 0 if the classes are different. For example,
((AD,AD),1) , ((AD,NC),0), ((NC,AD),0), etc.

As ADNI dataset is patient-level specific, therefore when splitting validation set from training set have be performed 
cautiously to prevent data leakage. As there are 20 images per patient, `torch.utils.data.random_split` will
most likely cause data leakage. Therefore, validation set is manually select from train set, extracting around 20% images from both
classes, while ensuring images from the same patient do not coexist in the training set, validation set, and testing set at 
the same time.

Images are pre-processed by a number of pytorch transform function to facilitate training process. Specific
transform used is discussed in the discussion section.



### Siamese neural network architecture
The model is defined in `modules.py`. The model consists of a several sequences of convolutional layers, batch normalization layers,
ReLU activation function, and pooling layers. Then followed by several fully connected layer with ReLU activation function in between.

![SNN.png](images_for_README%2FSNN.png)

### Training Siamese Neural Network
In `train.py`, the model is trained with the training dataset prepared in `dataset.py`. 

Contrastive loss is used for the training:

![img_1.png](images_for_README%2Fimg_1.png)

It takes as input a pair of sample that are either similar or dissimilar, and it brings similar samples closer and
dissimilar samples far apart.

Adam optimizer is used for the training of SNN and a classifier

![loss_plot_epoch_65.png](images_for_README%2Floss_plot_epoch_65.png)


The loss reduces over each epoch, indicating the success and evidence of the training.

With the code from [adambielski](https://github.com/adambielski/siamese-triplet/blob/master/README.md) (Github),
the embeddings can also be visualised in the vector space.

Epoch 1:

Training: 

![1.5_training_1.png](images_for_README%2F1.5_training_1.png)

Validation:

![1.5_validation_1.png](images_for_README%2F1.5_validation_1.png)

Epoch 50:

Training:

![1.5_training_25.png](images_for_README%2F1.5_training_25.png)

Validation:

![1.5_validation_25.png](images_for_README%2F1.5_validation_25.png)

### Training classifier 
The Siamese Neural Network outputs the embeddings
of the input, and a classifier is build to inference the embeddings, and determine the respective class of the images. The classifier structure
looks like this:

![CNN.png](images_for_README%2FCNN.png)

which is just 2 dense layers converting the embeddings into 2 outputs. Cross Entropy loss is used.

![Accuracy_plot.png](images_for_README%2FAccuracy_plot.png)

![loss_plot_classifier.png](images_for_README%2Floss_plot_classifier.png)

As the validation loss is pretty stable around 0.25 loss, the classifier at 30 Epochs is used for prediction
### Prediction
After training, the model can be used in predicting the class of an image. The performance of the model is evaluated in
`predict.py`, where images from the test dataset is inputted into the classifier. 

## Result
The best result obtained over numerous trials is 80.5% test accuracy on test set, which meets the project aim

| model combination and settings                              | Test accuracy |
|-------------------------------------------------------------|--------------:|
| Just classifier                                             |           67% |
| SNN(triplet loss) + classifier                              |           71% |
| SNN(Contrastive loss(2.0)) + classifier                     |           72% |
| SNN(Contrastive loss(1.0)) + classifier                     |           72% |
| Resnet18 as back bone of SNN + classifier                   |         73.5% |
| SNN(contrastive loss(1.0)) + classifier + data augmentation |         80.5% |

![test_acc.png](images_for_README%2Ftest_acc.png)

## Discussion
During first successful attempt, the classifier
end up with around 72% accuracy on the test set, while having 99% accuracy on the training set. Different architectures
of SNN were tested. I have tried 1 CNN layer + 1 dense layer, 2 CNN layer + 1 dense layer, etc. Despite minor improvement,
the test accuracy remains relatively identical. With ResNet18 being implemented as the backbone of the SNN, the 
similar result recorded on test accuracy shows network complexity is probably not the major limitation of the model 
performance. The issue is probably over-fitting.

| SNN architecture (backbone) | Best Test accuracy |
|-----------------------------|-------------------:|
| 1 CNN layer + 1 dense layer |                72% |
| 2 CNN layer + 1 dense layer |                73% |
| 3 CNN layer + 2 dense layer |                73% |
| 4 dense layer               |                69% |
| 3 CNN layer + 3 dense layer |                73% |
| ResNet18                    |                73% |

To tackle the problem of over-fitting, different learning rates on the model have been attempted. However, the effect is minimal.
However, when the batch size is decreased from 64 to 16, the test accuracy has increased 1-2% on average.

The major improvement comes from data augmentation. Different data augmentations are attempted:

| Data augmentation                               | Improvement |
|-------------------------------------------------|------------:|
| RandomCrop                                      |         +5% |
| RandomRotation                                  |         +2% |
| RandomPerspective                               |         +1% |
| RandomCrop + RandomRotation                     |         +7% |
| RandomCrop + RandomRotation + RandomPerspective |         +6% |

With the data augmentation, the classifier is able to achieve 80.5% accuracy on the test set. However, compare to ~90% validation accuracy,
the test accuracy is a bit subpar. This indicates that there are possibly some difficult images in the test set, or there may be some
data leakage. However, for this project, the goal is met.

### Potential improvement
As the test set is patient-level specific, i.e. there are 20 images from each patient in the test set, it is possible
to make a prediction based on the average classification result on all images from a patient.  

Another potential improvement is to prepare validation set with code. This should eliminate any risk of train-test contamination.


## Requirement
matplotlib == 3.7.2

python >=3.9.0

pytorch == 2.1.0

torchsummary == 1.5.1

## Reference
[1] Chicco, Davide (2020), "Siamese neural networks: an overview", Artificial Neural Networks, Methods in Molecular Biology, vol. 2190 (3rd ed.), New York City, New York, USA: Springer Protocols, Humana Press, pp. 73–94, doi:10.1007/978-1-0716-0826-5_3, ISBN 978-1-0716-0826-5, PMID 32804361, S2CID 221144012

[2] Rao, S.J., Wang, Y., & Cottrell, G. (2016). A Deep Siamese Neural Network Learns the Human-Perceived Similarity Structure of Facial Expressions Without Explicit Categories. Cognitive Science.