# Perceiver Transformer 
# Classify Alzheimer’s disease (normal and AD) of the ADNI brain data (see Appendix for link) using a visual or perceiver transformer [8, 9] set having a minimum accuracy of 0.8 on the test set. 
# Akshath Katyal
# 47144691


## Task Description:
The Perceiver is a cutting-edge deep learning architecture that has been making waves lately due to its exceptional performance across a diverse range of tasks. We decided to put it to the test ourselves and applied it to the problem of classifying Alzheimer's disease from brain images. With our implementation of the Perceiver model, we are confident in our ability to accurately classify Alzheimer's from non-invasive imaging methods, which can help with early detection and monitoring of the disease.


## Model/Algorithm Description
The Perceiver Transformer is a fresh deep learning structure that overcomes the usual constraint of Transformers requiring a fixed-input size. Its purpose is to tackle a vast range of input modes, from images to sound, without depending on domain-specific structures. The key perk of the Perceiver is its proficiency in handling inputs of any size by preserving a set of latent variables with a fixed size that interacts with the input data. The model was inspired by the tutorial from https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb which discusses how "Perceiver IO, landmark multimodal neural architectures that solve many of the issues with classic transformers". 



# List Dependencies
* Numpy version 1.26.1
* Pytorch 2.0
* Matplotlib 3.8.0


# Reproducibility 
1. In `dataset.py`, edit the path of your `AD_NC` dataset.
2. With the dependencies installed, run python train.py file 
3. Then run the predict.py to see results. 

# Results
This model was able to achieve really good results weirdly. It was no expected as the the number of epochs it was trained for was very small compared to larger numbers which often tend to get higher accuracy. The training time was pretty quick for 10 epochs and surprisingly showed 100% test set performance. This was definitely weird however when this model was run, it was not initially split into a validation set to compare it against. This could also be caused by data leakage and evaluation issues. 

![Training/Testing](<Training, Testing.png>)

Below are some example plots of the results
* Training accuracy over epochs went up to 100.
* Training loss went to 0. 
![1](<PerceiverTransform/Training Accuracy over Epochs.png>)
![2](<PerceiverTransform/Training Loss Over Epochs.png>)

I need to calculate the batch loss and run the test again with the validation set. 



# Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
* Preprocessing was done through Resizing of the images and normalizing the images using mean and standard deviation. 

* Justification of training and splitting of the data

When working with data, it's important to divide it into three parts. The biggest portion (65%) is used for training, allowing the model to learn the underlying patterns and features. A smaller subset (15%) is used for validation to adjust hyperparameters and prevent overfitting. This helps make informed decisions about the training process, such as when to stop early. The remaining 20% is reserved for testing, ensuring that the model's final evaluation is done on unseen data, providing an unbiased estimate of its generalization capabilities.

# References
* [1]: https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb
* [2]: https://medium.com/@curttigges/
