# Classifying Alzheimer's Disease Diagnoses Using Vision Trainsformer

This project aims to categorize the ADNI brain dataset into AD (Alzheimer's Disease) and NC (Normal Cognitive) groups. It employs a Vision Transformer network (ViT) based on the principles presented in the paper. The model was trained using an Adam Optimizer and the parameters were tweaked often to find a good accuracy. Each sample has 20 slices that is  240x256 greyscale image corresponding to a patient, which is to be classified as either NC or AD

## Dataset Splitting

The dataset is already split into 21,500 images for training, and 9000 images for testing. However, I needed a third split for validation in the dataset.py file.
data_val, data_test = random_split(TensorDataset(xtest, ytest), [0.7,0.3])
I used random_split for the validation and did a 70/30 split.
I then ended up with 6300 images for validation, and 2700 for testing.

## Preprocessing the data
The provided code preprocesses the image data by dividing it into patches, applying layer normalization and Multihead Attention mechanisms, and incorporating positional encoding before utilizing the Vision Transformer

## Training the data
These were the following parameters used for training. I didnt need a parameter for number of channels as we were only dealing with black and white data.
vit = VisionTransformer(input_dimen=128,
                        hiddenlayer_dimen=256,
                        number_heads=4,
                        transform_layers=4,
                        predict_num=2,
                        size_patch=(16,16))
input_dimen - Dimensionality of the input feature vectors to the Transformer
hiddenlayer_dimen - Dimensionality of the hidden layer in the feed-forward networks within the Transformer
number_heads - Number of heads to use in the Multi-Head Attention block
transform_layers - Number of layers to use in the Transformer
predict_num - Number of classes to predict
size_patch - Number of pixels that the patches have per dimension

The time taken to finish training depended on the parameters.
Using adam optimizer and learning rate = 1e-4 and 75 epoch, I had accuracy of 0.68 ( 5.5 hours )
With adamW optimizer and learning rate = 3e-4 and 100 epoch, I had a low accuracy of 0.53 ( 7 hours )

## Configuration 
All main configurations would be done in the train.py file
In the train function there is this: 
 optimizer = optim.AdamW(net.parameters(), lr=3e-4)
    epochs = 100
You can change between optimizers, learning rate and epoch value in here
Also in the end of the train.py file, there is the VIT.

vit = VisionTransformer(input_dimen=128,
                        hiddenlayer_dimen=256,
                        number_heads=4,
                        transform_layers=4,
                        predict_num=2,
                        size_patch=(16,16))




The project consists of four essential files, namely dataset.py, modules.py, train.py, and predict.py. The primary files to be executed are train.py and predict.py. The train.py file handles the training and testing of the model, allowing the option to save the model, along with recording the loss and validation accuracy for each epoch. This data is utilized by predict.py. Predict.py evaluates the actual output data as it can generate graphs depicting the loss and accuracy curves using the matplotlib library.



Key considerations: 
1. Inside the dataset.py file, script loads, preprocesses, and organizes medical image data from specific directories, converting the images to tensors, dividing them into training and testing sets with corresponding labels, and creating data loaders for training, testing, and validation.
2.  in train.py script imports required libraries, modules, and functions, then loads the data using returnDataLoaders from the dataset.py file. It defines an empty list for storing losses and accuracies, sets up a training function that utilizes the AdamW optimizer and CrossEntropyLoss
3. In the predict.py script, I plot two separate graphs. The first graph illustrates the accuracy vs epoch, displaying the trend of the model's accuracy over the training epochs. The second graph demonstrates the loss vs epoch, showcasing how the training loss varies throughout the training process.
4. The modules.py file contains functions and classes for implementing a Vision Transformer model, including an image patching function, an attention block class for multi-head attention, and a VisionTransformer class that applies linear transformations, positional embeddings