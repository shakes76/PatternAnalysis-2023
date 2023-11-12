# Visual Transformer 
#### Rhys Tyne s46481894

#### Task explanation
Problem 6: Classify Alzheimer's disease (normal or Alzheimer's Diseased) of the ADNI brain dataset using a visual transformer. 
The goal is to achieve a 0.8 accuracy on the test set.
This model aims to determine whether the given MRI brain scans belong to a person with a healthy or Alzheimer's 
diseased brain.

#### Dependencies
in dependencies.txt

#### Data Preprocessing
The data was preprocessed by the course staff so that most clinical information has been removed and only the images and
labels remain.
- training set is 21520 images
- testing set is 9000 images

The validation set is a subset of the training dataset, separated using the sklearn.model_selection.train_test_split() function.
The validation set is 0.15 of the training set which is 3228 images, generated randomly upon each time the model runs. 
After running the model a few times with different splits 0.1 and 0.2 it seems as this was the best split percentage I tested.
However, looking at the results it still seems to have over-fit to the training data as the testing accuracy was well 
below the training and validation accuracy.

Upon importation the image size of the data was resized to 128x128 from its original 256x240.

Data augmentation was used to artificially "increase the size" of the dataset by adding some randomness to the training set trhough use of 
RandomRotate, RandomZoom and RandomFlip in the keras.layers library.

An example input is shown in ![sample input]sample_input.jpeg and example output in ![sample output]exampleOutput.png.

#### Description of Model
The model used is a visual transformer (ViT) and the code used is based off of the keras example 
(https://keras.io/examples/vision/image_classification_with_vision_transformer/), which itself was based off of the paper 
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (https://arxiv.org/abs/2010.11929). 
Transformers were designed to solve problems where non-local correlation is present, most commonly seen in natural language
processes like ChatGPT, but have also been found to be very effective in image classification and segmentation problems.
Transformer models are built using a more generalised form of multi layer perceptron (mlp) which modifies its weights 
based on the input rather than keeping them fixed.
Transformer models use multi-head self-attention to split the image into a series of patches (thereby maintaining some 
amount of positional information) and these patches are then processed by the encoder. Because attention is learnt and 
does not use recursion they can obtain similar results to other models like RNN's but faster. The basic architecture of 
a transformer model is ![transformer Architecture]transformerArchitecture.jpg, which shows how the patches are created
and then flattened and used as input for the transformer.

This model uses BinaryCrossEntropy to measure loss in the model, which is a good choice for a ViT 2-class classification 
problem. 



#### Results
Testing accuracy was 68.12% which was well below the accuracy which both the training validation accuracy which reached
a peak of 86% and 92% respectively. 
This suggests that the model may have been over-fitting to the training data as can be seen in 
![model accuracy]modelAccuracy.png.
Additionally, you can see the model performance start to plateau at around epoch 80, which aligns with when the 
reduceLRonPlateau function starts working, this is indicative that the reduction factor was too high or the patience was 
too low.

The loss in the model, see ![model loss]modelLoss.png, tells a similar story to that of the accuracy as it starts to 
right around the 80 epoch mark as well. This is interesting as these two metrics are largely uncorrelated and indicates 
the model made "quite large" mistakes.