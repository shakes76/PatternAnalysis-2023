# Siamese Network Introduction

The purpose of this report is to employ the Siamese network deep learning model with the ANDI dataset for the classification of normal and Alzheimer's disease (AD) cases. The Siamese network was chosen over other classification models like traditional neural networks and random forests due to its unique capability to measure similarity and dissimilarity between data points. This enables it to effectively differentiate between normal brain scans and those affected by Alzheimer's disease, making it an ideal choice for this particular task.

Classifying brain scans using a Siamese network holds significant importance in the field of medical diagnostics and neurology. It offers several notable advantages, including improved accuracy in identifying Alzheimer's disease 
at an early stage, leading to better treatment outcomes and enhanced patient care. By learning the subtle differences between normal and AD-affected brain scans, Siamese networks play a crucial role in assisting medical professionals in their diagnosis and decision-making processes.

Moreover, the use of Siamese networks in brain scan classification contributes to reducing the societal and economic burdens associated with AD. Early detection can lead to more effective interventions, potentially slowing down the progression of the disease and reducing healthcare costs. Additionally, it allows for the development of more precise and personalized treatment plans, improving the quality of life for affected individuals and their families.

Siamese network-based brain scan classification also has implications in the field of neuroscience research. It aids in the study of the disease's progression, providing valuable insights for researchers and helping to identify potential biomarkers for Alzheimer's disease. Furthermore, it facilitates the development of advanced diagnostic tools and predictive algorithms, which can be instrumental 
in large-scale epidemiological studies and drug development efforts.

In conclusion, the utilization of Siamese networks for the classification of brain scans, particularly for Alzheimer's disease detection, has the potential to revolutionize the field of neurology and medical diagnostics. It empowers healthcare professionals with a powerful tool to enhance their diagnostic accuracy and provides hope for more effective treatments and interventions. 
Furthermore, it contributes to the global effort to combat Alzheimer's disease by aiding research and enabling early interventions, ultimately improving the lives of those affected by this devastating condition.

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