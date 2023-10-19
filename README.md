# Visual Transformer 
#### Rhys Tyne s46481894

#### Task explanation
Problem 6: Classify Alzheimer's disease (normal or Alzheimer's Diseased) of the ADNI brain dataset using a visual transformer. 
The goal is to achieve a 0.8 accuracy on the test set.

#### Dependencies
in dependencies.txt

#### Data Preprocessing
The data was preprocessed by the course staff.
- training set is 21520 images
- testing set is 9000 images

The validation set is a subset of the training dataset, separated using the sklearn.model_selection.train_test_split() function.
The image size of the data was resized to 128x128.

Data augmentation was used to increase the size of the dataset by adding some randomness to the training set trhough use of 
RandomRotate, RandomZoom and RandomFlip in the keras.layers library.

An example input is shown in sample_input.jpeg

#### Description of Model
The model used is a visual transformer (ViT) and the code used is based off of the keras example 
(https://keras.io/examples/vision/image_classification_with_vision_transformer/). The model aims to determine whether 
the given MRI brain scans belong to a person with a healthy or Alzheimer's diseased brain.

Transformers were designed to solve problems where non-local correlation is present, for exmaple natural language
processes, but have also been found to be very effective in image classification problems. Transformer models use multi-head 
self-attention to split the image into a series of patches (thereby maintaining some amount of positional information)
and these patches are then processed by the encoder. 

#### Results
Testing accuracy was 68.12% which was well below the accuracy which both the training and validation datasets achieved. 
This suggests that the model may have been overfitting to the training data as can be seen in accuracy1.png.
The learning rate decreased due to the ReduceLRonPlateau at epochs 74, 83, 92 and 99 reaching a minimum of 1.6e-6.




