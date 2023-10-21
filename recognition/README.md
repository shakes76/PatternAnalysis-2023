# Visual Transformer (ViT) for classifying Alzheimer's Disease

## OVERVIEW:
This project is dedicated to creating a machine learning model for the classification of Alzheimer's disease (AD) and normal brain scans, employing advanced Visual or Perceiver Transformer models. The primary objective is to achieve a minimum accuracy of 0.8 on the test dataset
n 2020, the groundbreaking paper titled "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" demonstrated that traditional Convolutional Neural Networks could be surpassed by Vision Transformers. These Vision Transformers proved capable of delivering outstanding results when compared to state-of-the-art convolutional networks, all while demanding fewer computational resources for training.The adoption of Vision Transformers in this project is driven by the potential to harness their efficiency and accuracy in medical image classification, ultimately contributing to the advancement of Alzheimer's disease diagnosis and enhancing the healthcare landscape.

![visual transformer](https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/579168d1-8dbe-4177-a549-52b8a930319c)

## MODEL ARCHITECTURE:
The Visual Transformer (ViT) is an influential neural network architecture for computer vision, adapting the Transformer model from natural language processing to process images. ViT divides images into patches, transforms them into vectors, and employs a multi-head self-attention mechanism to capture complex spatial relationships. It uses stacked layers of self-attention and feedforward networks to extract features and make predictions. ViT excels in tasks like image classification and object detection, thanks to its ability to handle global and local information. However, training ViT models usually requires pre-training on large image datasets due to its high parameter count, yet it has significantly impacted the field of computer vision.

The Vision Transformer model consists of the following steps:
1.Split an image into fixed-size patches
2.Linearly embed each of the patches
3.Prepend [class] token to embedded patches
4.Add positional information to embedded patches
5.Feed the resulting sequence of vectors to a stack of standard Transformer Encoders
6.Extract the [class] token section of the Transformer Encoder output
7.Feed the [class] token vector into the classification head to get the output

## Transformer Encoder: 
The Transformer Encoder is composed of two main layers: Multi-Head Self-Attention and Multi-Layer Perceptron. Before passing patch embeddings through these two layers, we apply Layer Normalization and right after passing embeddings through both layers, we apply Residual Connection.

 ![image](https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/871fe1ac-dd5c-408a-a6f4-71afb08b3fde)

## Dependencies:

1.Python 3.10.4

2.Tensorflow 2.10.0: An open-source machine learning framework.

3.Tensorflow Addons 0.18.0: An extension library for TensorFlow, providing additional functionalities.

4.Matplotlib 3.5.2: A data visualization library used for creating plots and charts in Python.

5. Keras 2.0.8: A high-level neural networks API that runs on top of TensorFlow.

## Repository Overview:
parameters.py: Stores hyperparameters for model configuration.
modules.py: Contains the Vision Transformer's fundamental components.
dataset.py: Manages data loading functions.
train.py: Compiles and trains the model with relevant functions.
predict.py: Enables model predictions with its functions.

## Alzheimer's Disease Neuroimaging Initiative(ADNI) Dataset:
The dataset comprises 21,500 grayscale images, each with dimensions 256x240, divided into 21,500 training samples and 9,000 test samples. These images fall into two distinct categories: Alzheimer's Disease (AD) patient brain scans and those representing normal cognitive condition (NC).
1.	Training Set: 21,520 images 
2.	Validation Set: 4500 images 
3.	Testing Set: 4500 images 

<img width="128" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/c193a68a-fd6a-45f9-bef9-018d42d91eb6">    <img width="128" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/2f92c46a-534f-4089-9e6f-5fce21b77471">

## RESULTS:
The current best model trained reached 57.63% accuracy on the test set, with 60.52% training accuracy. This was achieved using the following parameters:

IMAGE_SIZE = 128
PATCH_SIZE = 8
BATCH_SIZE = 16
EPOCHS = 10
PROJECTION_DIM = 128 
LEARNING_RATE = 0.001
ATTENTION_HEADS = 5
DROPOUT_RATE = 0.2
TRANSFORMER_LAYERS = 5 
WEIGHT_DECAY = 0.001
MLP_HEAD_UNITS = [256, 128] 

1.	IMAGE_SIZE: The dimensions (in pixels) to which the input images are resized.
2.	PATCH_SIZE: The size of the image patches used as input to the Vision Transformer.
3.	BATCH_SIZE: The number of data samples processed in each training iteration.
4.	EPOCHS: The number of times the entire dataset is processed during training.
5.	PROJECTION_DIM: The dimensionality of the projected feature vectors in the model.
6.	LEARNING_RATE: The rate at which the model adjusts its parameters during training.
7.	ATTENTION_HEADS: The number of self-attention heads used in the model's multi-head self-attention mechanism.
8.	DROPOUT_RATE: The probability of dropout (disabling neurons) in the model to prevent overfitting.
9.	TRANSFORMER_LAYERS: The number of stacked layers in the Vision Transformer.
10.	WEIGHT_DECAY: A regularization term to control the model's complexity and prevent overfitting.
11.	MLP_HEAD_UNITS: The sizes of the feedforward neural network layers in the model.

The accuracy and loss plots are displayed below:

### Accuracy Plot:
1.	In the early stages of training, accuracy is low as the model learns to distinguish between the two classes.
2.	As training progresses, the accuracy for the training set and validation set increases. A rising trend indicates that the model is improving its ability to make correct predictions.
3.	The rise and then drop in validation accuracy suggest the model might be overfitting.
4.	Then subsequent increase in validation accuracy is promising, indicating that the model could be recovering from overfitting.

<img width="240" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/ce35238b-e0eb-41b3-925c-41c1fb0755dd">

### Loss Plot:
1.	The loss plot displays the value of the loss function over training epochs.
2.	Loss measures the difference between the predicted and actual class labels. 
3.	Initially, the loss is typically high, and then it gradually decreases as the model learns to make better predictions. A consistent downward trend in loss signifies improved model performance. However, the loss is gradually decreasing for the training set but for the validation set we see a increasing pattern which can be a strong indicator of overfitting.

<img width="268" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/3464f7ce-5780-4d51-99e8-dd1b591cb2dc">

## CONCLUSIONS:
Model Improvement Efforts:
I tried various methods to enhance our model's performance, including adjusting learning rates, weight decay, image sizes, batch sizes, and patch sizes. Unfortunately, none of these adjustments led to substantial improvements. Both the validation and testing accuracies remained below 80% throughout much of the training process.

Challenges in Achieving 80% Accuracy:
The original goal was to attain over 80% accuracy on the test data, but this proved to be quite challenging. It would have required meticulous tweaking of numerous model settings to see improvements.

Potential Enhancements:
To enhance this project and get closer to the 80% accuracy target, I suggest two key approaches:

1.	Larger Dataset: Using a more extensive dataset could be highly beneficial. More data often leads to better model performance.
2.	Pre-trained Vision Transformers: Following a strategy similar to the one outlined in the original Vision Transformer paper, we could start with pre-trained Vision Transformers and then fine-tune them with our specific dataset. This approach has the potential to significantly improve the model's accuracy.
   

## REFERENCES:
Dosovitskiy A. et al, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Vaswani A. et al, Attention is All You Need

Vision Transformer in TensorFlow

Image classification with Vision Transformer

https://keras.io/examples/vision/image_classification_with_vision_transformer/

https://github.com/shakes76/PatternFlow/blob/topic-recognition/recognition/46965611-ADNI-Brain-Visual-Transformer/README.md

https://github.com/nerminnuraydogan/vision-transformer

https://arxiv.org/abs/2112.13492



