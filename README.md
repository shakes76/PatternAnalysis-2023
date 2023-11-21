# Description

This contains methods to create embeddings via a Triplet Siamese Network, which can then be used to create a classifier for Alzheimer's disease using the ADNI dataset. The model attempts to identify Alzheimer's from MRIs, which would be very helpful for radiology.

# Algorithm

The triplet Siamese model works by taking in both a positive and negative input, as well as an anchor. An identical backbone (in this case, resnet) is used on all three inputs to produce an embedded space. The aim of the model is to make the distance between the positive and anchor less than the distance between the negative and the anchor, without knowing which camp the anchor belongs to. The lost function used to achieve this subtracts the negative distance from the positive to ensure a lower loss when closer to positive. This creates a good embedded space to then use to classify inputs. Using this, individual outputs from the embedded space can be inputted into a Random Forest Classifier, which can then be used to identify Alzheimers.

# Pre-Processing

Training Images are randomly rotated between -5 and 5 degrees, offset between -5% and 5% and scaled between 95% and 105%. They are then intensity normalized. After that, the mean and standard deviation is found for the entire training set, and both the training and testing sets are normalized using that mean and standard deviation. Finally, the training set is split into two, with 80% being used for the Triplet Siamese Model and the remaining 20% for the classifier.

# Triplet Siamese Model

Each epoch, each batch is randomly sorted to have a different triplet each time. The inputs then undergo the rotation, offset and scaling transform from pre-processing to keep the model on it's toes. The loss function is calculated using a tiny amount of L2 regularization, 0.0001. This helps with the generalization of the model.

The learning rate begins very low at 0.0001, gradually increases to a normal level of low, 0.001, which is reached halfway through training. After the peak is reached, the learning rate decreased at the same rate back to its starting level.

The optimizer used is a Scholastic Gradient Descent optimizer. When looking into the optimizer I discovered that weight decay and L2 regularization are the same thing, so actually 0.0005 is also used.

# Classification Model

The embedded space used for the three inputs in the Triplet Siamese Model is then used inside another model to classify individual inputs. The classification model is a Random Forest model, with 600 estimators, 250 min split and max depth of 12. This allows for more generalization and beats Neural Networks by quite a margin.

# Dependencies

Python version: 3.10.12

Pytorch version: 2.1.0, Cuda 11.8

Numpy version: 1.23.5

Matplotlib version: 3.7.1

Sklearn version: 1.2.2

PIL version: 9.4.0

# Inputs and Outputs

![image](https://raw.githubusercontent.com/FinnRobertson15/PatternAnalysis-2023/topic-recognition/inputs.png)

Top: 1 (Alzheimer's)

Bottom: 0 (No Alzheimer's)

# Results

![image](https://raw.githubusercontent.com/FinnRobertson15/PatternAnalysis-2023/topic-recognition/PCA.png)

Triplet loss graph

![image](https://raw.githubusercontent.com/FinnRobertson15/PatternAnalysis-2023/topic-recognition/train.png)

Classification Accuracy of 65%