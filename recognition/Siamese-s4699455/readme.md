# Siamese Network Introduction
    The algorithm of this project is to solve the problem of similarity between AD and NC CT scan images.
    The Siamese network contains two identical feature extraction networks (VGG16),
    Calculate the L1 distance between the two eigenvalues, and finally use the full connection and sigmod function 
    to fit the similarity of the two pictures.
    
    In the training part, the data set preparation needs to be in the form of a pair, including {AD, AD}, {NC, NC}, 
    {AD, NC}, and the corresponding labels are 1, 1, and 0 respectively.
    For the reasoning part, the input needs to be in the form of pairs of two pictures, and the output is 
    the similarity of the two pictures.

# Data preprocessing part:
## Data clipping
    The original training data ADNC data set is a single-channel grayscale image of 256X240 size.
    The effective information of the original data is concentrated in the middle of the picture. Use the crop operation 
    on the picture to remove 16 pixels from the top, bottom, left and right.
    The final model input size is 224X208X1, which reduces the amount of calculation.
## Data normalization   
    All input data are normalized and sent to the network, effectively improving the training convergence speed.

# Script description
## 1、run_train.sh
    Parallel training using a single machine with multiple cards.
## 2、kill_proc.sh
    Delete zombie processes that terminate abnormally during training.

# Code file description
## 3、train.py
    Training file, called by run_train.sh.
## 4、predict_begin.py
    Inference startup file, reads data, initializes the network, and outputs the similarity of two pictures.
## 5、predict.py
    Inference files, data preprocessing, model inference, result comparison.

## 6、nets/modules.py
    Siamese network results, after using the backbone of vgg16 to extract features,
    Use the feature values of the two images as the L1 distance and output the predicted value.
## 7、nets/vgg.py
    vgg16 basic network, used to extract features.

## 8、utils/dataset.py
    Read the data set, prepare the input data into pairs, read the image, crop, normalize and send it to the network.
## 9、utils/utils_aug.py
    Data augmentation (not used yet)
## 10、utils/utils_fit.py
    Called by train.py, it traverses the batch data and performs forward propagation and directional propagation.
## 11、utils/utils.py
    Other related tools

# Model, log, training data file description
## 12、 log
    pth --- model file
    log --- log file
    loss --- tensorbord format training file

## 13、data set directory
    AN_NC
## 14、environment.yaml
    conda configuration environment