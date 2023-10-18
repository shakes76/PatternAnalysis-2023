# Siamese Network Introduction

This project showcases the application of a Siamese Network in the context of Alzheimer's Disease detection, employing deep learning techniques. The Siamese Network is specifically crafted to compare pairs of medical images and determine whether they pertain to the same category, such as Alzheimer's Disease or Cognitive Normal. The primary objective remains achieving an accuracy of around 0.8 on the test dataset. This README document offers a concise outline of the project's structure, instructions for utilization, and insights into the functionalities of each project component.

# Dependencies

To run the code，you need the following dependencies:

Python 3.6.13

Pip 21.2.2

Matplotlib 3.7

NumPy 1.24

PyTorch 2.0

# Usage

## Run the model on the ADNI dataset:

Importing the necessary datasets and modules can be achieved by importing the functions and classes defined in `dataset.py` and `modules.py`.
Run the `train.py` script to train the Siamese network. Make sure the dataset is ready and pass the data to the training script.

## Convert image file paths into image and label pairs:

If you need to convert a list of image file paths into image and label pairs, you can use the `_convert_path_list_to_images_and_labels` function in `predict.py`. You need to use this function to load the image, do the preprocessing and return the image pair and its label.

## Calculate similarity and accuracy:

Use the detect_image function in `predict.py` to calculate the similarity of the model on the test set and obtain the accuracy of the model. This helps understand how the model performs on test data. There is also `log\predict.py` that can generate images about the loss to better understand and monitor the performance of the model.

# Data Preprocessing

## Data clipping

The original training data ADNC data set is a single-channel grayscale image of 256X240 size.
The effective information of the original data is concentrated in the middle of the picture. Use the crop operation 
on the picture to remove 16 pixels from the top, bottom, left and right.
The final model input size is 224X208X1, which reduces the amount of calculation.

## Data normalization   

All input data are normalized and sent to the network, effectively improving the training convergence speed.

# Project Structure
## train.py

Training file, called by run_train.sh.

## predict_begin.py

Inference startup file, reads data, initializes the network, and outputs the similarity of two pictures.

## predict.py

Inference files, data preprocessing, model inference, result comparison.

## nets/modules.py

Siamese network results, after using the backbone of vgg16 to extract features,
Use the feature values of the two images as the L1 distance and output the predicted value.

## nets/vgg.py

vgg16 basic network, used to extract features.

## utils/dataset.py

Read the data set, prepare the input data into pairs, read the image, crop, normalize and send it to the network.

## utils/utils_aug.py

Data augmentation (not used yet)

## utils/utils_fit.py

Called by train.py, it traverses the batch data and performs forward propagation and directional propagation.

## utils/utils.py

Other related tools

## log

pth --- model file

log --- log file

loss --- tensorbord format training file

# How It Works

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*23mikUF3HBJGUqrX7tMKQQ.png)

**Architecture:** Siamese networks usually consist of two identical subnetworks (twin networks) that share parameters and weights. Each subnetwork accepts an input sample and maps it to a fixed-length vector (often called an embedding vector). These two sub-networks process two input samples respectively and generate their embedding vectors.

**Shared weights:** The two sub-networks in the Siamese network have the same structure and parameters, which means they perform the same feature extraction operation.

**Loss function:** The goal of the Siamese network is to make the distance between similar samples in the embedding space smaller, while the distance between dissimilar samples in the embedding space is larger. To achieve this goal, a contrastive loss function is often used, such as cosine cosine loss (contrastive loss). These loss functions encourage the distance between embedding vectors to meet specific conditions in order to better distinguish between similar and dissimilar samples.

**Training:** The Siamese network uses pairs of input data during training, including similar sample pairs and dissimilar sample pairs. The parameters of the network are trained via backpropagation and gradient descent to minimize the loss function.

**Inference:** Once the Siamese network is trained, new samples can be mapped to the embedding space and the distance between them calculated to determine their similarity.

# Results

## **Inputs**

The ADNI dataset provides a pair of brain images, one from an AD patient and one from a healthy individual.

## Outputs

![](https://raw.githubusercontent.com/SKY-YY88/PatternAnalysis-2023/Siamese-network/recognition/Siamese-s4699455/out__%5B0.9096765518188477%5D.png)

![](https://raw.githubusercontent.com/SKY-YY88/PatternAnalysis-2023/Siamese-network/recognition/Siamese-s4699455/out__%5B0.012563883326947689%5D.png)

 Accuracy: 0.91

## Plots

![https://raw.githubusercontent.com/SKY-YY88/PatternAnalysis-2023/Siamese-network/recognition/Siamese-s4699455/log/epoch_loss.png](https://raw.githubusercontent.com/SKY-YY88/PatternAnalysis-2023/Siamese-network/recognition/Siamese-s4699455/log/epoch_loss.png)

# References

G. Koch, R. Zemel, R. Salakhutdinov et al., “Siamese neural networks for one-shot image recognition,” in ICML deep learning workshop, vol. 2. Lille, 2015, p. 0

# Author

Qianchen Zhao

# License

Apache License 2.0