# Visual Transformer (ViT) for classifying Alzheimer's Disease

## OVERVIEW:
This project is dedicated to creating a machine learning model for the classification of Alzheimer's disease (AD) and normal brain scans, employing advanced Visual or Perceiver Transformer models. The primary objective is to achieve a minimum accuracy of 0.8 on the test dataset
n 2020, the groundbreaking paper titled "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" demonstrated that traditional Convolutional Neural Networks could be surpassed by Vision Transformers. These Vision Transformers proved capable of delivering outstanding results when compared to state-of-the-art convolutional networks, all while demanding fewer computational resources for training.The adoption of Vision Transformers in this project is driven by the potential to harness their efficiency and accuracy in medical image classification, ultimately contributing to the advancement of Alzheimer's disease diagnosis and enhancing the healthcare landscape.

![visual transformer](https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/579168d1-8dbe-4177-a549-52b8a930319c)

## MODEL ARCHITECTURE:
The Visual Transformer (ViT) is an influential neural network architecture for computer vision, adapting the Transformer model from natural language processing to process images. ViT divides images into patches, transforms them into vectors, and employs a multi-head self-attention mechanism to capture complex spatial relationships. It uses stacked layers of self-attention and feedforward networks to extract features and make predictions. ViT excels in tasks like image classification and object detection, thanks to its ability to handle global and local information. However, training ViT models usually requires pre-training on large image datasets due to its high parameter count, yet it has significantly impacted the field of computer vision.

## Vision Transformer Model Steps

1. **Split Image into Patches:** Divide the input image into fixed-size patches.

2. **Linearly Embed Patches:** Embed each of the patches into lower-dimensional vectors using linear projections.

3. **Prepend [class] Token:** Add a special [class] token to the embedded patches, representing the class label.

4. **Add Positional Information:** Incorporate positional information into the embedded patches to maintain spatial context.

5. **Feed to Transformer Encoders:** Feed the resulting sequence of vectors into a stack of standard Transformer Encoders, which perform self-attention and feedforward operations.

6. **Extract [class] Token Output:** Isolate the output corresponding to the [class] token from the Transformer Encoder.

7. **Classification Head:** Pass the [class] token vector through a classification head to obtain the final output, typically representing class predictions or scores.


## Transformer Encoder: 
The Transformer Encoder is composed of two main layers: Multi-Head Self-Attention and Multi-Layer Perceptron. Before passing patch embeddings through these two layers, we apply Layer Normalization and right after passing embeddings through both layers, we apply Residual Connection.

<img src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/871fe1ac-dd5c-408a-a6f4-71afb08b3fde" width="800" />


## Dependencies:


| Dependency        | Version    | Description                                       |
|-------------------|------------|---------------------------------------------------|
| Python            | 3.10.4     | Programming language used for the project.       |
| TensorFlow        | 2.10.0     | Open-source machine learning framework.           |
| TensorFlow Addons | 0.18.0     | Extension library for additional TensorFlow functionalities. |
| Matplotlib        | 3.5.2      | Data visualization library for creating plots and charts. |
| Keras             | 2.0.8      | High-level neural networks API running on top of TensorFlow. |

## Repository Overview:
## Repository Overview

| Module             | Description                                                    |
|--------------------|----------------------------------------------------------------|
| parameters.py      | Stores essential hyperparameters for model configuration.      |
| modules.py         | Contains core components of the Vision Transformer (ViT).     |
| dataset.py         | Manages data loading functions and dataset preparation.        |
| train.py           | Compiles and trains the ViT model with relevant functions.    |
| predict.py         | Enables model predictions with its functions.                 |


## Alzheimer's Disease Neuroimaging Initiative(ADNI) Dataset:
The dataset comprises 21,500 grayscale images, each with dimensions 256x240, divided into 21,500 training samples and 9,000 test samples. These images fall into two distinct categories: Alzheimer's Disease (AD) patient brain scans and those representing normal cognitive condition (NC).
| Dataset          | Number of Images | Categories                        |
|------------------|------------------|----------------------------------|
| Training Set     | 21,520           | AD Patient, NC (Normal Cognitive) |
| Validation Set   | 4,500            | AD Patient, NC                   |
| Testing Set      | 4,500            | AD Patient, NC                   |


<table>
    <tr>
        <td style="text-align: center;">
            <img width="256" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/c193a68a-fd6a-45f9-bef9-018d42d91eb6">
            <br>
            <span>AD</span>
        </td>
        <td style="text-align: center;">
            <img width="256" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/2f92c46a-534f-4089-9e6f-5fce21b77471">
            <br>
            <span>NC</span>
        </td>
    </tr>
</table>



## RESULTS:
The current best model trained reached 57.63% accuracy on the test set, with 60.52% training accuracy. This was achieved using the following parameters:

### Model Configuration


| Configuration         | Value      | Description                                           |
|-----------------------|------------|-------------------------------------------------------|
| IMAGE_SIZE            | 128        | Dimensions (in pixels) to which input images are resized. |
| PATCH_SIZE            | 8          | Size of image patches used as input to the Vision Transformer. |
| BATCH_SIZE            | 16         | Number of data samples processed in each training iteration. |
| EPOCHS                | 10         | Number of times the entire dataset is processed during training. |
| PROJECTION_DIM        | 128        | Dimensionality of projected feature vectors in the model. |
| LEARNING_RATE         | 0.001      | Rate at which the model adjusts its parameters during training. |
| ATTENTION_HEADS       | 5          | Number of self-attention heads in the multi-head self-attention mechanism. |
| DROPOUT_RATE          | 0.2        | Probability of dropout to prevent overfitting. |
| TRANSFORMER_LAYERS    | 5          | Number of stacked layers in the Vision Transformer. |
| WEIGHT_DECAY          | 0.001      | Regularization term to control model complexity. |
| MLP_HEAD_UNITS        |            | Sizes of feedforward neural network layers in the model. |
|                       | Layer 1:   | 256 units |
|                       | Layer 2:   | 128 units |



The accuracy and loss plots are displayed below:

### Accuracy Plot:

The accuracy plot visualizes the model's performance during training:

- In the early stages of training, accuracy is low as the model learns to distinguish between the two classes.
- As training progresses, both training set and validation set accuracy increase, indicating the model's improved ability to make correct predictions.
- The initial rise followed by a drop in validation accuracy suggests the possibility of overfitting.
- Subsequent increase in validation accuracy is promising, indicating the model might be recovering from overfitting.

<img width="512" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/ce35238b-e0eb-41b3-925c-41c1fb0755dd">

In the plot, you can observe the accuracy trends for the training and validation sets, which provide insights into the model's learning process and potential overfitting.

### Loss Plot:

The loss plot tracks the performance of the model's loss function during training:

- Loss measures the difference between the predicted and actual class labels.
- Initially, the loss is typically high, and then it gradually decreases as the model learns to make better predictions.
- A consistent downward trend in loss signifies improved model performance on the training set.
- However, if the loss is gradually decreasing for the training set while increasing for the validation set, it can be a strong indicator of overfitting.

<img width="512" alt="image" src="https://github.com/saakshigupta2002/PatternAnalysis-2023/assets/62831255/3464f7ce-5780-4d51-99e8-dd1b591cb2dc">

In the plot, you can observe the loss trends for both the training and validation sets, providing insights into the model's learning process and the potential occurrence of overfitting.


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



