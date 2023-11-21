# Perceiver Transformer 
# Classify Alzheimer’s disease (normal and AD) of the ADNI brain data (see Appendix for link) using a visual or perceiver transformer [8, 9] set having a minimum accuracy of 0.8 on the test set. 
# Akshath Katyal
# 47144691

## Task Information:
The Perceiver is a cutting-edge deep learning architecture that has been making waves lately due to its exceptional performance across a diverse range of tasks. We decided to put it to the test ourselves and applied it to the problem of classifying Alzheimer's disease from brain images consisting of both (AD or Alzheimers disease and NC or Normal cognition). This implementation of the Perceiver model is used to accurately classify Alzheimer's from non-invasive imaging methods, which can help with early detection and monitoring of the disease.-

## Model/Algorithm Description
The Perceiver Transformer is a fresh deep learning structure that overcomes the usual constraint of Transformers requiring a fixed-input size. Its purpose is to tackle a vast range of input modes, from images to sound, without depending on domain-specific structures. The key perk of the Perceiver is its proficiency in handling inputs of any size by preserving a set of latent variables with a fixed size that interacts with the input data. The model was inspired by the tutorial from https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb which discusses how "Perceiver IO, landmark multimodal neural architectures that solve many of the issues with classic transformers" but was modified accordingly for the ADNI dataset. 

# List Dependencies
* Numpy version 1.26.1
* Pytorch 2.0
* Matplotlib 3.8.0
* Python 3.11.4

# Reproducibility 
1. In `train.py`, edit the path of your `AD_NC` dataset to be utilized by the loaders.
2. With the dependencies installed, run python train.py file 
3. This will show you the results based on epoch, once the training is done will display two visualizations of it.
4. Then run the `predict.py`.
5. Input the path of any image from the test dataset to check if it is able to predict the image correctly.

# Results
At first This model was able to achieve really good results weirdly. It was no expected as the the number of epochs it was trained for was very small compared to larger numbers which often tend to get higher accuracy. It was later discovered that the dataset was not being read correctly and the labelling of the images were wrong displaying 1 type of labels everytime hence showing 100% accuracy. The training time was pretty quick for 10 epochs and surprisingly showed 100% test set performance. However, after the model was fixed with correct paths, data split and labelling as well as changing images not from RGB but leaving it as grayscale. This was able to fix the issue however when this model was run, the test accuracy was around 50% (50.44% for 10 epochs and batch size of 5). This is equivalent to random guessing between two types (pretty much 50/50). 

Here is an image of the training and accuracy of the data and test set. 
![(<Training, Testing.png>)](<PerceiverTransform/Test Set Accuracy .png>)

Below are visualized plots of the loss and training accuracy over epochs.
* Previously before the model was modified, Training accuracy after 1 epochs went up to 100%.
* Training loss went to 0. 

![1]![(<PerceiverTransform/Training Accuracy over Epochs.png>)](<PerceiverTransform/Training accuracy over Epochs.png>)
![2]![(<PerceiverTransform/Training Loss Over Epochs.png>)](<PerceiverTransform/Training loss over Epochs.png>)

Based on these results, the loss and training accuracy it can be interpreted that the model is not properly trained. Increasing the number of epochs and decreasing the learning rate only changes the results by bare minimum. 


# Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
* Preprocessing was done through Resizing of the images to fit typical neural network input sizes and then converting into tensors 
* Justification of training and splitting of the data: 

When working with data, it's important to divide it into three parts. The biggest portion (65%) is used for training, allowing the model to learn the underlying patterns and features. A smaller subset (35%) is used for validation to adjust hyperparameters and prevent overfitting. This helps make informed decisions about the training process, such as when to stop early, providing an unbiased estimate of its generalization capabilities.

Potential change in the splitting can effect the model learning better however, the issue seems to be with the training. 

# References: 
* [1]: https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb
* [2]: https://github.com/clint-kristopher-morris/Perceiver-Transformer/tree/main

