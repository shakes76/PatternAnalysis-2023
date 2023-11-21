# Classificaion of Alzheimer’s disease with Siamese network

This project shows how a Siamese network can be used to determine the similarity between two images of Alzheimer’s disease. 

## Siamese network basics

In the early 1990s, the Siamese network was presented, which can solve the signature verification problem. The model consists of two identical neural networks. This means that the weights and the number of different layers are the same for the left and right neural networks. At the end of this network, they are connected, and the L1 norm or L2 norm is calculated as distance. This layer is also called the distance layer. After the distance layer, the network uses a dense layer with a sigmoid function. The output of this dense layer is the result. The aim of this network is to determine the similarity between two input images.
An example image of the Siamese network is shown in the image below:

![SaimeseNetwork](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/SaimeseNetwork.PNG)


## Dataset

For this project, the ADNI dataset is used. This dataset consists of two classes and is split into training and testing data. The train directory consists of 10400 images with the class AD and 11120 with the class NC. This dataset is divided into training and validation datasets. In this project, the ratio is 80% training and 20% validation. Furthermore, the data are split so that all slices of brain images from one brain are either in the training or validation dataset. The test dataset comprises 4460 AD images and 4540 NC images. All the images can be downloaded from the Alzheimer’s Disease Neuroimaging Initiative website (https://adni.loni.usc.edu/).


The images below show an example of the AD and NC. The frist image is a AD image and the second is a NC image.
![AD](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/AD.jpeg)
![NC](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/NC.jpeg)

In this project, a data generator is used, which creates image pairs with the matching label. There are four different image pair combinations: AD, AD with the label 1, AD, NC with the label 0, NC, NC with the label 1 and NC, AD with the label 0. The data generator which creates the training data also shuffles the data. However, the data generator which creates the validation pairs does not shuffle the data. This way the neural network can be tested with the same image pairs during the training. The data generator uses a batchsize of 32 which results into 128 image pairs for on batch. Always 32(AD,AD), 32(AD,NC), 32(NC,NC) and 32(NC,AD) images.


## Siamese network model

This project uses a VGG16 with only one dense layer for the left and right side of Siamese neural networks. The VGG (Visual Geometry Group) was discovered at the Department of Engineering Science, University of Oxford. The paper about this neural network introduces different sizes of the model. The VGG16, for instance, has 16 convolution layers. In addition, after two or three convolution layers, a max pooling layer is added. After the 16 convolution layers, three dense layers are used. 
In the project, after the VGG, a dense layer and a distance layer are used, which calculates the L1 norm. At the end, a dense layer calulates how similar the two images are. The fully connected layer from the VGG16 at the end and the fully connected layer at the end of the whole network use a sigmoid activation function.

The image below shows the structure of the network:
![Model](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/Model.PNG)


## Training of the model

For the training of the model, a maximum of 50 epochs are trained. However, callbacks are used, which allows us to get the best validation loss. In addition, the callback function stops the training if the validation loss is no longer improved. For the loss function, contrastive loss or triplet loss can be used. In this project, contrastive loss is used. For the optimizer, Adam is used with a learning rate of 0.0001. The training of the neural network is done with Google Colab Pro. The V100 is used as a GPU.
The training of the model is shown below:

![Training](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/Training.PNG)

The image shows how fast the accuracy increases and the loss function decreases per epoch. This can also be shown as graphs. The figures below show the graphs of the training accuracy, training loss, validation accuracy, and validation loss.

![AccuracyTrainImage](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/AccuracyTrainImage.png)

![LossTrainImage](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/LossTrainImage.png)

The graphs show that the model overfits because the training accuracy reaches nearly one after ten epochs. Different numbers of dropout layers or batch normalization layers are tried out to avoid this. However, I couldn't prevent that, so I finally left them out.
The accuracy of the validation data stops heavily increasing at around 0.7. The loss function graph shows that the training loss decreases to nealy 0 and the validaiton loss settles at approximately 0.2.

## Testing

For the testing of the neural network, two functions are used. The first function is the evaluation function from TensorFlow. The model reaches an accuracy of around 0.6 with the test dataset. The test dataset is created with the validation generator function, generates the image pairs with the matching label, and uses the test images.
The figure below shows the result.

![Test1](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/Test1.PNG)

The other test uses my own created test loop. This function first loads 32 images from the AD and NC test data. Creates 128 image test pairs and makes a prediction with the prediction function. Afterwards, the prediction is evaluated, and if the value is smaller than 0.5, the prediction label is 0. In addition, if the value is bigger than 0.5, the prediction label is 1. These prediction labels are compared with the actual labels and count how often the model is correct or incorrect. In the end, the accuracy is calculated, and some image pairs with the prediction are plotted as an example. The image below shows the accuracy of the test function is 0.6.

![Test2](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/Test2.PNG)

The plotted image pair result shows a correctly predicted image pair and an incorrectly predicted image pair.

![PlotPrediciton](https://github.com/UQpfister/PatternAnalysis-2023/blob/topic-recognition/recognition/DanielPfisterSiameseNetwork/Images/PlotPrediciton.PNG)

As a result, the project doesn't reach the goal of 0.8 accuracy. I tried different learning rates and convolutional networks, such as the resnet18 however, it was impossible to achieve more than 0.6 accuracy. I think it has to do something with the overfitting of the training.


## Using the code

To use the code, you have to change the different folder paths of the dataset. Moreover, the model weight's saving and loading path must also be modified. The project consists of four different Python files where the model is trained with the train.py. The Python files predict.py allow us to test the model. The Python files dataset.py and modules.py contain functions which are required in the train.py and predict.py files. The ColabSaimeseNetwork.ipynb file is used to train the model in Google Colab.

## References
[1] G. Koch, R. Zemel, and R. Salakhutdinov, Siamese neural networks for one-shot image recognition, Jul 2015

[2] K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, Apr 2015

[3] Image similarity estimation using a Siamese Network with a contrastive loss (https://keras.io/examples/vision/siamese_contrastive/)
