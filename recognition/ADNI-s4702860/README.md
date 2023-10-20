# Classifier based on Siamese Network
## Model Architecture and Problem
Siamese neural networks have multiple inputs each trained on the same model sharing weights, 
and as such are exceptional at being able to predict if two objects are the same class. 

Triple input siamese networks by taking inputs, and outputting the relative embeddings. 

From this a classifier can be created on these embeddings. This idea is what has been explored in this report. 

## Installation
To run this prgram, a few dependencies will need to be installed. Each of these can be 
installed via the commandline. Below shows the proper commands plus a brief explanation:

### 1. Install TensorFlow
This project relies on TensorFlow for neural network creation and as such requires tensorflow. 
This can be installed via the command prompt using:
```
pip install tensorflow
```
### 2. Install MatPlotLib
This program outputs several visualisation plots and thus requires Matplotlib. 
```
pip install matplotlib
```
### 3. Install scikit-learn
This program uses scikit-learn in order to determine the accuracy of the classifier. 
```
pip install scikit-learn
```
### 4. TensorFlow AddOns
TripleSemiHardLoss was used for the training of the neural network. 
```
pip install tensorflow-addons
```
### 5. UMAP
UMAP has been used in order to determine the class regions
```
pip install umap-learn
```


## How to run
Before running any programs, make sure the file path for the train and test set 
in dataset.py is correct before continuing. The assumption is that "AD_NC"
is in the current working direction. 

If this is not where the data is loaded, simply open dataset.py, and
change the path variable in the two generators. 

There are two files which need to be ran:
```
train.py
```
Program that trains both the siamese neural network and the classification model. 

Run this program. Note, the data loader is quite computationally expensive and as such 
the runtime for just loading the data is approximately 10 minutes. 

Comparitivly, 1000 worked on my machine however may not for other peoples. 

```
predict.py
```
Program that runs based upon saved weights (done in train.py). 

Upon running this program, the weights created in predict.py will be loaded into a model, 
and various plots will be displayed to show the effect of using the Siamese model.

## Preprocessing and Data Augumentation
Image data generators were chosen as not only are they computationally cheap, they also allow
for data Augumentation quite easily. 
Typically the data was normalized, however, it also allowed for random variences such as pizel dimension, rotation etc.

Attempts were made to try and load the data generator into the model. Results found that the model
while training on the dataset would take substancial time per iteration. 
As such, only a subset of features were trained and tested upon. 

## Example Use
The main task of this is to attempt to classify the data as either AD (Alzeimers disease) or NC 
(normal cognition). To classify this data, first:
### 1. Load Anchor Embeddings
First, start by creating a model with the same weights as siamese_model.h5. 

```
base_model = load_base()
classifier = load_classifier()
test_embeddings = base_model.predict(test_anchor)

```
After getting the test embeddings, you are able to get the predicted class by:
```
predictions = classifier.predict(test_embeddings)

```