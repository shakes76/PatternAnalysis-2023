# Siamese model

### Description
The Siamese model is a neural network that work's on two given input's to create comparable outputs. These output's can be used to judge the similarities between given inputs. Being able to compare the similarities of two input's is very useful, and Siamese network's can be used for:
- Facial recognition
- Signature analysis
- Matching queries with documents

This Siamese network is used to solve the issue of identifying Alzheimers within the brain. Using the dataset provided by the **ADNI** (Alzheimers disease neuroimaging initiative). It take's an image from one of the two classes, normal and Alzheimers disease, Then selects a 2nd image from one of the two classes and attempts to find if they are from the same class or not. By learning features in both a normal brain and one with Alzheimers disease, it can learn to spot certain similarities and differences between the two classes. This can then hopefully be used to train a model that can produce an accuracy of 80%

### How the algorithm works
Before we jump into how the algorithm works and some of the components in the model, first I will list imported and downloaded modules and libraries:
- Pytorch (version 2.1)
- Cuda (version 11.8)
- Numpy (version 1.24.0)
- data (link : https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)
- Matplotlib (version 3.7)
