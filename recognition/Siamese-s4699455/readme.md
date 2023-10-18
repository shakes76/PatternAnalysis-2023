# Siamese Network Introduction

This project showcases the application of a Siamese Network in the context of Alzheimer's Disease detection, employing deep learning techniques. The Siamese Network is specifically crafted to compare pairs of medical images and determine whether they pertain to the same category, such as Alzheimer's Disease or Cognitive Normal. The primary objective remains achieving an accuracy of around 0.8 on the test dataset. This README document offers a concise outline of the project's structure, instructions for utilization, and insights into the functionalities of each project component.

# Data Preprocessing
## Data clipping

The original training data ADNC data set is a single-channel grayscale image of 256X240 size.
The effective information of the original data is concentrated in the middle of the picture. Use the crop operation 
on the picture to remove 16 pixels from the top, bottom, left and right.
The final model input size is 224X208X1, which reduces the amount of calculation.

## Data normalization   

All input data are normalized and sent to the network, effectively improving the training convergence speed.

# Script Description
## 1、run_train.sh

Parallel training using a single machine with multiple cards.

## 2、kill_proc.sh

Delete zombie processes that terminate abnormally during training.

# Project Structure
## 3、train.py

Training file, called by run_train.sh.

## 4、predict_begin.py

Inference startup file, reads data, initializes the network, and outputs the similarity of two pictures.

## 5、predict.py

Inference files, data preprocessing, model inference, result comparison.

## 6、nets/modules.py

Siamese network results, after using the backbone of vgg16 to extract features,
Use the feature values of the two images as the L1 distance and output the predicted value.

## 7、nets/vgg.py

vgg16 basic network, used to extract features.

## 8、utils/dataset.py

Read the data set, prepare the input data into pairs, read the image, crop, normalize and send it to the network.

## 9、utils/utils_aug.py

Data augmentation (not used yet)

## 10、utils/utils_fit.py

Called by train.py, it traverses the batch data and performs forward propagation and directional propagation.

## 11、utils/utils.py

Other related tools

# Model, log, training data file description
## 12、 log

pth --- model file
log --- log file
loss --- tensorbord format training file

## 13、data set directory

AN_NC

## 14、environment.yaml

Conda configuration environment

# How It Works

![](C:\Users\47647\OneDrive\Documents\WeChat Files\wxid_d2ktwuhr1vnk12\FileStorage\Temp\09f78597018ebc121ab86ae46ef4135.png)

**Architecture:** Siamese networks usually consist of two identical subnetworks (twin networks) that share parameters and weights. Each subnetwork accepts an input sample and maps it to a fixed-length vector (often called an embedding vector). These two sub-networks process two input samples respectively and generate their embedding vectors.

**Shared weights:** The two sub-networks in the Siamese network have the same structure and parameters, which means they perform the same feature extraction operation.

**Loss function:** The goal of the Siamese network is to make the distance between similar samples in the embedding space smaller, while the distance between dissimilar samples in the embedding space is larger. To achieve this goal, a contrastive loss function is often used, such as cosine cosine loss (contrastive loss). These loss functions encourage the distance between embedding vectors to meet specific conditions in order to better distinguish between similar and dissimilar samples.

**Training:** The Siamese network uses pairs of input data during training, including similar sample pairs and dissimilar sample pairs. The parameters of the network are trained via backpropagation and gradient descent to minimize the loss function.

Inference: Once the Siamese network is trained, new samples can be mapped to the embedding space and the distance between them calculated to determine their similarity.

# Training Processes

**Data preparation:** Prepare a training data set containing pairs of data, where each pair consists of two examples, usually a "positive" pair and a "negative" pair. Positive sample pairs represent similar examples, while negative sample pairs represent dissimilar examples.

**Build a Siamese network:** Define and build a Siamese neural network architecture that consists of two identical subnetworks ("twin networks") that share weights during training. These two sub-networks are used to process each input example.

**Siamese network output:** Each sub-network maps input examples to feature vectors or embedding spaces. These feature vectors can be vectors with higher dimensions, representing the features of the input.

**Similarity measure:** Measures the similarity of two examples by calculating the distance or similarity measure between their feature vectors, such as Euclidean distance or cosine similarity.

**Loss function:** Define a loss function, usually contrastive loss, to guide model training. The goal of this loss function is to make the similarity score of positive sample pairs higher than the similarity score of negative sample pairs.

**Model training:** Using the training data set, the model optimizes the loss function through backpropagation and gradient descent algorithms. The goal is to adjust the weights of the Siamese network so that it can better distinguish between positive and negative pairs.

**Validation and tuning:** Use a validation dataset to evaluate the performance of your model. Based on the validation results, the model can be fine-tuned for better performance.

**Testing and Deployment:** Once the model is trained and validated, it can be used for testing and deployment to perform actual similarity comparison tasks.

# Results

## **Inputs**

The ADNI dataset provides a pair of brain images, one from an AD patient and one from a healthy individual.

## Outputs

![https://github.com/SKY-YY88/PatternAnalysis-2023/blob/Siamese-network/recognition/Siamese-s4699455/out__%5B0.012563883326947689%5D.png]()

![](https://raw.githubusercontent.com/SKY-YY88/PatternAnalysis-2023/Siamese-network/recognition/Siamese-s4699455/out__%5B0.012563883326947689%5D.png)

 Accuracy: 0.91

## Plots

![](C:\Users\47647\AppData\Roaming\Typora\typora-user-images\image-20231018163807680.png)

# Dependencies

To run the code，you need the following dependencies:

Python 3.6.13

Pip 21.2.2

Matplotlib 3.7

NumPy 1.24

PyTorch 2.0

# References

G. Koch, R. Zemel, R. Salakhutdinov et al., “Siamese neural networks for one-shot image recognition,” in ICML deep learning workshop, vol. 2. Lille, 2015, p. 0