# Classification of Alzheimer’s Disease using Siamese Neural Network
(A Deep Learning Approach with Application to the ADNI Dataset) <br>
Name - Danita Anubhuti Prakash <br>
Student ID: s4745511

## 1.Overview
The focus of this project is to develop medical image classification system using Siamese networks. Here, I used the ADNI dataset which contains brain images divided into two categories: Alzheimer's disease (AD) vs. normal (NC). This project can contribute significantly to early Alzheimer's disease diagnosis. The code in this repository is designed for building, training, and evaluating two neural network models: a Siamese Network and a Classification Model. These models are typically used for image-based tasks like similarity comparison and classification. The Siamese Network learns to extract meaningful features from images and computes a similarity score between pairs of images. The Classification Model, built on top of the Siamese Network, is used for classifying images based on the features extracted by the Siamese Network.

## 2. Data - ADNI Dataset 
### 2.1 Data Collection
The dataset used for this project was obtained from a particular URL: "<https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download>". I used HTTP GET request to downloaded the dataset (from the mentioned link). The dataset was saved locally in a specified file path after successful download. The downloaded content was written to this local file and was ensured by the code. The downloaded zip file was extracted to reveal its contents subsequently. 

I then did som preprocessing to understand the dataset:
The following are the example images under the two categories as given below (from the training set):


![image](https://github.com/danitaanubhuti/PatternAnalysis-2023/assets/52007397/5ed9e1cd-57d5-4bb5-aa89-b54fb18c553f)

The code for data collection and preprocessing is mentioned in utils.py
### 2.1 Data Pairing
The preparation of data for training a Siamese neural network is streamlined by the siamese_data_loader function in the dataset.py. The consistency and readiness of the input data for neural network training is ensured by this process. The creation of image pairs is the core of the function and a fundamental aspect of Siamese network training. Pairs of images from the same class ("AD" or "NC") and pairs from different classes is constructed by it while assigning corresponding labels to indicate whether the pairs match (label 1) or not (label 0). There are four different combination of images that can be created:
1. Poisitve Pair(1) - (AD, AD)
2. Poisitve Pair(1) - (NC, NC)
3. Negative Pair(0) - (AD, NC)
4. Negative Pair(0) - (NC, AD)

The function deliberately randomizes the order of images within each class to eliminate any potential biases prior to pairing. It shuffles the dataset after pairing, ensuring that the order of pairs is randomized. Based on a user-defined ratio, the final dataset is split into training and validation sets and is then batched for efficient model training. This function simplifies the process of preparing data for Siamese neural network training, in summary, emphasizing the randomization of data before pairing and shuffling after pairing to promote unbiased and effective training.

## 3.Working of Siamese on the ADNI Dataset
Two interconnected models are used to classify images as either 0 - "normal" or 1 - "Alzheimer's disease" by using. These are the Siamese Neural Network (SNN) and a classification model. Here's how these components work together to achieve this task:

### 3.1 Siamese Neural Network (SNN):
The SNN plays an essential role in learning and representing the similarity between pairs of images. It comprises of a network that processes individual images and computes feature vectors for them. The SNN architecture is designed to generate feature vectors. These feature vectors effectively capture the essential characteristics of the input images. The SNN's subnetwork learns to differentiate between image pairs belonging to different classes (0 or 1) by producing distinct feature vectors for them, for this binary classification problem. To quantify their similarity, the distance between these feature vectors is computed using a Euclidean distance layer. Smaller distances indicate more similar images whereas larger distances represent dissimilar pairs.sIn the Siamese Network both neural network branches share the same set of neural network weights.  Each branch processes one input image and learns a common representation for both inputs.

The architecture of Siamese Network can be explained with the following diagram:
![image](https://github.com/danitaanubhuti/PatternAnalysis-2023/assets/52007397/484ba153-e239-4e72-8dea-e5475ece5233)

The Siamese architecture revolves around a shared subnetwork, for my project I choose a convolutional neural network (CNN). The CNN's goal is to extract features suitable for conducting similarity comparisons. 
Following feature extraction, each input image is mapped to a feature vector that embodies the underlying characteristics of the images. The feature vectors from both input images are then used to calculate a similarity or dissimilarity metric. 

The network architecture includes an output layer, which is a single neuron with a sigmoid activation function. This is used to interpret the computed similarity metric. In similarity-based tasks, the output neuron's value represents the estimated similarity between the input data points. Training a Siamese network typically involves using a contrastive loss or margin-based loss function. This loss function encourages the network to minimize the distance between feature vectors for similar pairs of data and maximize it for dissimilar pairs. The Siamese network is trained using pairs of data samples with known similarity labels. During training, the network learns to map similar data pairs closer in feature space and dissimilar data pairs farther apart. This learned similarity metric can be applied to various applications, enabling the network to make similarity assessments even when only relative similarity information is available.

### 3.2 Loss, Distance Metric and Optimizer
Contrastive loss is a common choice for training Siamese networks, in particular, when the goal is to learn embeddings or representations for similarity-based tasks. Similar pairs of data points are encouraged to be closer to each other by the loss in the embedding space while pushing dissimilar pairs farther apart. This loss function drives the Siamese network to learn embeddings that are useful for distinguishing between similar and dissimilar pairs of data. By minimizing the contrastive loss, the network learns to create embeddings in which similar data points are clustered together and dissimilar data points are pushed apart. 

I used the Euclidean distance as the distance metric for my model, between two input vectors, represented by tensors. It computes the squared differences, summing them, and taking the square root. This distance measurement is a floating-point value, ensuring accurate vector dissimilarity assessment.

![image](https://github.com/danitaanubhuti/PatternAnalysis-2023/assets/52007397/dfc0633b-4b94-49ea-a3c1-10083aebd1da)



**Loss Calculation:** The Siamese network generates embeddings for both data points for each pair of data points. The Euclidean distance between these embeddings is calculated by the loss function. The similarity or dissimilarity between the two data points are in the embedding space is quantified by this distance. Using this distance the contrastive loss is computed. If the distance between similar data points is too large or if the distance between dissimilar data points is too small, it penalizes the model. In specific, the loss term for a pair of similar data points is proportional to the square of the distance between their embeddings, aiming to minimize this distance. In contrast, the loss term for a pair of dissimilar data points is proportional to the square of the maximum difference between a specified margin and the distance between their embeddings.

![image](https://github.com/danitaanubhuti/PatternAnalysis-2023/assets/52007397/1d8d2333-985b-47c7-b98b-529714c18496)


Lastly, for the optimizer I use the Adam Optimizer as the optimizer for my model. The learning_rate parameter specifies the step size at which the optimizer updates the model's parameters during training. In this case, you've set it to 0.0001. A smaller learning rate means smaller steps, which can help the optimizer converge more stably but may require more training epochs. The choice of the learning rate depends on the specific problem, and it often requires experimentation to find the optimal value. A common range for learning rates is between 0.1 and 0.0001.

### 3.3 Classification Model:
The core of this project lies in the feature extraction, as realized through the create_cnn_network function. This function defines and constructs a Convolutional Neural Network (CNN) model . It operates under the fundamental premise that images are composed of hierarchical patterns and features. To capture these details in these images, the function takes as input the essential dimensions of the input images—height and width. The architecture consits of an input layer, ensuring that the neural network can seamlessly interpret the data structure. The architectural framework extends to include convolutional layers (conv1, conv2, and conv3) and pooling layers that help in the process of feature extraction. These convolutional layers employ Rectified Linear Unit (ReLU) activation functions and adeptly applied padding to preserve the spatial richness of the image. Between each convolutional layer, max-pooling layers (pool1, pool2, and pool3) are introduced to reduce the dimensionality within the image data. At this point, the feature maps are flattened into a one-dimensional vector, setting the stage for the subsequent processing. The processing continues as the function ushers these flattened features through two fully connected layers (dense1 and dense2) characterized by ReLU activation functions and the infusion of L2 kernel regularization. These theoretical principles fecilitates in the creation of the model.

The project employs the create_classification_model function for binary classification tasks. This function takes the feature extraction subnet created by create_cnn_network and utilizes it for classification. The function commences by creating an input layer designed to accommodate grayscale images sized at 128x128 pixels. These input images pass through the subnet—the feature extraction component—to derive feature vectors that encapsulate essential image characteristics. To ensure the robustness and consistency of these feature vectors, batch normalization is applied. Lastly, a final output layer with a sigmoid activation function is introduced, which provides probability estimates regarding the input's classification into class 1, where values close to 1 signify a positive classification, and values close to 0 indicate a negative classification. The classification model is compiled with an Adam optimizer using binary cross-entropy loss, with accuracy as the evaluation metric, making it ready to serve in binary classification tasks based on feature representations extracted by the subnet.

### 3.4 Training and Classification:
Feeding labeled image pairs into the SNN and the classification model is involved in the training process. The classification model is optimized to make accurate predictions based on the feature vectors from the SNN during training.

The classification model can take pairs of new, unlabeled images and predict the probability of these images belonging to class 1, effectively classifying them as Alzheimer's disease or normal after training.Using backpropagation and optimization techniques the models are fine-tuned to minimize the classification error. 

In summary, the SNN's ability to learn image representations that emphasize the similarity between pairs of images, especially for those of the same class is leveraged by this workflow. Binary predictions are then made by the classification model using these representations, allowing it to classify images as 0 (normal) or 1 (Alzheimer's disease) based on their similarity. Effective image classification is achieved by this approach while benefiting from the deep learning capabilities of neural networks.

## 4.Model – 
### 4.1 Model Architecture
**The network is designed with two identical sub-networks, each processing a different input sample with the same weights. The name “Siamese” comes from this fact.** In order to generate a prediction, the outputs from these two sub-networks are then compared in the final layer.

For tasks such as image similarity and classification, The Siamese Neural Network (SNN) implemented in this project is a powerfully designed architecture. The network core consists of three main components: a subnetwork, a distance layer, and a classification layer. The subnetwork is constructed using Convolutional Neural Network (CNN) layers and acts as the feature extractor for input images. It comprises three convolutional layers, each of which is followed by max-pooling operations, creating an effective hierarchy for feature extraction. Two fully connected (dense) layers with Rectified Linear Unit (ReLU) activation functions and L2 kernel regularization are added to prevent overfitting and ensure robustness. The subnetwork serves as the foundation for the network's feature representation and contributes significantly to the overall model's performance.

The Euclidean difference between feature vectors extracted by the sub network for two input images is calculated by the distance layer. This difference, is an essential component of the network's design as it represents the similarity or dissimilarity between the images. The network utilizes a contrastive loss function to enhance training and classification accuracy, which factors in the difference between predicted and ground truth labels. The classification layer incorporates the subnetwork's output and also leverages batch normalization for normalization purposes. It concludes the network's architecture with a dense layer using sigmoid activation which facilitates binary classification. The modified Siamese network excels in tasks where similarity and classification of images are paramount and overall it showcases a sophisticated yet intuitive architecture.

### 4.2 Step by Step working
The first subnetwork takes an image (input1) as input and passes through convolutional layers and fully connected layers, after which we get a vector representation of the image. Again the second image (input2) is passed through a network that is exactly the same with the same weights and parameters. Two encodings output_vector1 and output_vector2 from the respective images are obtained, we can compare these two to know how similar the two images are. If the encodings are similar then the images will also be quite similar. The distance between these two vectors will be measured. If the distance between these is small then the vectors are of the same classes or **similar** and if the distance between is larger then the vectors are **different** from one another, based on the score.

The diagrammatic representation can be shown as:
![image](https://github.com/danitaanubhuti/PatternAnalysis-2023/assets/52007397/1435dcc6-e722-4397-afd4-5dfbd3291695)


Ultimately, the output of my Siamese Network is a similarity score. For various tasks, such as determining whether two images belong to the same category or classifying whether they are similar or dissimilar, this score can be used. It provides a way to measure the likeness between images. 

## 5.Project Content
The following are the python files used :
1. "modules.py": I've implemented the architecture of the Siamese network, the contrastive loss function, and other related functionsin this file.

2. "dataset.py":This file contains file which help to load the data for training and classification.

3. "train.py": This script is responsible for training, validating, testing, and saving my deep learning model. I've imported the model architecture from "modules.py" and the data loader from "dataset.py." I've also included code for plotting the losses and metrics during training to provide a visual understanding of the model's performance.

4. "predict.py": In this file, I've demonstrated how to use the trained model for making predictions. 

5. "utils.py": This file contains all the extra codes I have used in process to do the project (data collection and preprocessing).

The execution of the project starts with training the model using the train.py file following which the predictions can be done using the predict.py file.
Add the paths links are changed from the original to keep the paths in my computer private.

## 5.1 Sequence of Working of the Code
Training the Siamese Network is done using trainSNN function. It loads the training and validation data, trains the Siamese Network, and saves the trained model to a file.Next, the train the Classification Model is carried out using the trainClassifier function. It loads the Siamese model, extracts the shared subnetwork, creates a classification model, and trains it on classification data. Then visualizations are represented to understand the training progress of both models.

SNN_PATH and CLASSIFIER_PATH are file paths used for saving trained machine learning models in the Hierarchical Data Format (HDF5) format. In the provided code, these paths are used to save the trained Siamese Network (SNN) model and the Classification Model (Classifier) after training. This allows you to persist the trained models so that you can later load and use them for making predictions without retraining the models each time.

To make predictions, I use the predict.py to load the trained Classification Model (saved as Classifier.h5) and evaluates it on test data. It also provides an example of printing predictions and actual labels. The plot_training_history function generates two subplots for each model's training history. The first subplot displays training accuracy and validation accuracy, with different colors to distinguish them. The second subplot illustrates the training loss with another color. This function helps monitor the performance and progress of both models during training.

## 6. Results (Training and Testing)
In the training process, the Siamese Network (SNN) underwent 40 epochs, while the Classification Model was trained for 20 epochs. For the SNN, it started with a loss of 5.0073 and an accuracy of 52.55%, gradually improving with each epoch to achieve a final loss of 0.2049 and an accuracy of 65.87%. The Validation loss improved from 2.0056 to 0.1979, and the Validation accuracy increased from 59.28% to 66.25%. Similarly, the Classification Model started with a loss of 0.4172 and an accuracy of 81.55%, reaching a final loss of 0.0753 and an accuracy of 98.57%. The Validation loss decreased from 0.4413 to 0.1039, and the Validation accuracy increased from 87.92% to 97.29%. These results indicate successful training, with both models improving their performance significantly. 

An 80-20 split, with 80% of the data allocated to training and 20% to validation, strikes a balance between having ample training data for model learning and a substantial validation set for fine-tuning. This ensures the model's robustness while efficiently using the available data.

![image](https://github.com/danitaanubhuti/PatternAnalysis-2023/assets/52007397/cd43d515-fc7c-4ef7-a896-f656e40bafb1)

In the evaluation results, the model achieved a loss of approximately 1.97 and an accuracy of about 66.36% (the accuracy fluctuates between 58 - 66 when multiple training and predictions are done). The subsequent predictions and actual labels show the model's performance on individual data points. The model predicted the likelihood of a binary outcome (0 or 1) for each data point, and the actual labels indicate the true values. The predictions are continuous values between 0 and 1, representing the model's confidence in its predictions. For example, a prediction close to 0 indicates low confidence in a positive outcome, while a prediction close to 1 indicates high confidence in a positive outcome. The model's performance is evaluated based on its ability to accurately classify data points into the correct categories.

## 7. Dependencies:
1. TensorFlow (version 2.6.0)
2. Keras (version 2.6.0, which is often integrated with TensorFlow)
3. Matplotlib (version 3.4.3)
4. NumPy (version 1.21.3)
5. os (standard python library)
6. random (standard python library)

## References
1.	<https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513>
2.	<https://www.baeldung.com/cs/siamese-networks>
3.	<https://builtin.com/machine-learning/vgg16>
