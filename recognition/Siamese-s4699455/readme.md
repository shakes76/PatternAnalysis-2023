## 1、 nets/modules.py
    Siamese network results, after using the backbone of vgg16 to extract features,
    Make the L1 distance between the feature values of the two images and output the predicted value.
## 2、nets/vgg.py
    vgg16-basic network, used to extract features

## 3、utils/dataset.py
    Read the data set, prepare the input data into pairs, read the image, 
    crop, normalize and send it to the network
## 4、utils/utils_aug.py
    Data augmentation (not used yet)
## 5、utils/utils_fit.py
    Called by train.py, it traverses the batch data and 
    performs forward propagation and directional propagation.
## 6、utils/utils.py
    Other related tools