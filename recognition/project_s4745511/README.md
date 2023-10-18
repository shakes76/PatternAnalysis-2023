# Using Siamese Neural Network for Enhanced Classification of Alzheimer’s Disease in Brain Imaging Data
(A Deep Learning Approach with Application to the ADNI Dataset)
## 1.Overview
The development of a medical image classification system using Siamese networks is focused on, in this project. Classifying medical images into two categories: Alzheimer's disease (AD) vs. normal (NC) is the primary goal, on the dataset used. Patient outcomes can be improved in medical imaging applications, as this project can contribute significantly to early disease diagnosis.
## 2.Project Content
You will find several key files and directories that make up the framework for my deep learning project, as I’ve structured this project:
1. **"modules.py":** The source code of various components of the deep learning model is present in this file. I've implemented the architecture of the Siamese network, the contrastive loss function, and other related functions or classes, in this file.
2. **"dataset.py":** The data loader and preprocessing functions needed to load and prepare my dataset is included in this file. Data augmentation, image preprocessing, and other data-specific tasks are incorporated.
3. **"train.py":** This script is responsible for training, validating, testing, and saving my deep learning model. I've imported the model architecture from "modules.py" and the data loader from "dataset.py." I've also included code for plotting the losses and metrics during training to provide a visual understanding of the model's performance.
4. **"predict.py":** In this file, I've demonstrated how to use the trained model for making predictions. This comprises loading the model, preprocessing input data, and obtaining predictions. I've provided examples of printing out the results or visualizations showcasing the model's predictions wherever applicable.
5. **"utils.py":** I aim to make it more accessible and maintainable by organizing the project in this structured manner. The execution of the project starts with training the model using the train.py file following which the predictions can be done using the predict.py file

## 3.Model – 
### 3.1 Model Architecture
**The network is designed with two identical sub-networks, each processing a different input sample with the same weights. The name “Siamese” comes from this fact.** In order to generate a prediction, the outputs from these two sub-networks are then compared in the final layer.

![image](https://github.com/me50/s-prak/assets/116443738/283b3975-f830-4327-8355-0ddb76c3e65a)

For tasks such as image similarity and classification, The Siamese Neural Network (SNN) implemented in this project is a powerfully designed architecture. The network core consists of three main components: a subnetwork, a distance layer, and a classification layer. The subnetwork is constructed using Convolutional Neural Network (CNN) layers and acts as the feature extractor for input images. It comprises three convolutional layers, each of which is followed by max-pooling operations, creating an effective hierarchy for feature extraction. Two fully connected (dense) layers with Rectified Linear Unit (ReLU) activation functions and L2 kernel regularization are added to prevent overfitting and ensure robustness. The subnetwork serves as the foundation for the network's feature representation and contributes significantly to the overall model's performance.

The Euclidean difference between feature vectors extracted by the sub network for two input images is calculated by the distance layer. This difference, is an essential component of the network's design as it represents the similarity or dissimilarity between the images. The network utilizes a contrastive loss function to enhance training and classification accuracy, which factors in the difference between predicted and ground truth labels. The classification layer incorporates the subnetwork's output and also leverages batch normalization for normalization purposes. It concludes the network's architecture with a dense layer using sigmoid activation which facilitates binary classification. The modified Siamese network excels in tasks where similarity and classification of images are paramount and overall it showcases a sophisticated yet intuitive architecture.

### 3.2 Step by Step working
The first subnetwork takes an image (A) as input and passes through convolutional layers and fully connected layers, after which we get a vector representation of the image. Again the second image(B) is passed through a network that is exactly the same with the same weights and parameters. Two encodings E(A) and E(B) from the respective images are obtained, we can compare these two to know how similar the two images are. If the encodings are similar then the images will also be quite similar. The distance between these two vectors will be measured. If the distance between these is small then the vectors are of the same classes or **similar** and if the distance between is larger then the vectors are **different** from one another, based on the score.

The diagrammatic representation can be shown as:
![image](https://github.com/me50/s-prak/assets/116443738/80c3d6aa-201d-47df-91e0-c74f1fb2daf8)

## 4.Working of Siamese on the ADNI Dataset
To classify images as either 0 (e.g., "normal") or 1 (e.g., "Alzheimer's disease") by training two interconnected models is the aim of the code and training workflow: a Siamese Neural Network (SNN) and a classification model. Here's how these components work together to achieve this classification task:

### 4.1 Siamese Neural Network (SNN):
The SNN plays an essential role in learning and representing the similarity between pairs of images. It comprises of a subnetwork that processes individual images and computes feature vectors for them. The SNN architecture is designed to generate feature vectors. These feature vectors effectively capture the essential characteristics of the input images.
The SNN's subnetwork learns to differentiate between image pairs belonging to different classes (0 or 1) by producing distinct feature vectors for them, for this binary classification problem. To quantify their similarity, the distance between these feature vectors is computed using a Euclidean distance layer. Smaller distances indicate more similar images whereas larger distances represent dissimilar pairs.

#### Architecture:
Both neural network branches share the same set of neural network weights in the Siamese Network.  Each branch processes one input image and learns a common representation for both inputs. Key to the network's ability to compare images rather than classify them is that they are trained separately on different inputs.

The architecture of my Siamese Network consists of the following components:
![image](https://github.com/me50/s-prak/assets/116443738/6408942b-56d5-4614-b070-3d25c65af433)

**Convolutional and Pooling Layers:** Extracting meaningful features from the image samples of each network is done by these layers. To capture patterns and details in the input images, they apply filters. This results in a set of feature maps that represent the images.

**Comparison Layer:** This is the last layer in each branch. It generates an embedding, which is a compact representation of the data sample. The design of this layer can vary based on the specific task. For example, you're using a custom distance layer that computes the Euclidean difference between the feature vectors of the input images.

**Comparison Function:** To produce a prediction that determines how similar or different the two data samples (input images) are, the Siamese Network uses a comparison function. In my code, I use this comparison function to calculate the similarity score, which indicates the degree of similarity between the input images.

Ultimately, the output of my Siamese Network is a similarity score. For various tasks, such as determining whether two images belong to the same category or classifying whether they are similar or dissimilar, this score can be used. It provides a way to measure the likeness between images. It is applicable in various applications, including image similarity comparison, face recognition, and more.

### 4.2 Loss and Distance Metric
Contrastive loss is a common choice for training Siamese networks, in particular, when the goal is to learn embeddings or representations for similarity-based tasks, such as image similarity or face recognition. Similar pairs of data points are encouraged to be closer to each other by the loss in the embedding space while pushing dissimilar pairs farther apart. When you want to create embeddings that enable you to measure the similarity or dissimilarity between data points, this concept is crucial.

This loss function drives the Siamese network to learn embeddings that are useful for distinguishing between similar and dissimilar pairs of data. By minimizing the contrastive loss, the network learns to create embeddings in which similar data points are clustered together and dissimilar data points are pushed apart. This is a fundamental concept for various tasks like face recognition, image similarity, and, in your project, Alzheimer's disease classification.

**Loss Calculation:** The Siamese network generates embeddings for both data points for each pair of data points. The Euclidean distance between these embeddings is calculated by the loss function. The similarity or dissimilarity between the two data points are in the embedding space is quantified by this distance. Using this distance the contrastive loss is computed. If the distance between similar data points is too large or if the distance between dissimilar data points is too small, it penalizes the model. In specific, the loss term for a pair of similar data points is proportional to the square of the distance between their embeddings, aiming to minimize this distance. In contrast, the loss term for a pair of dissimilar data points is proportional to the square of the maximum difference between a specified margin and the distance between their embeddings. The embeddings of dissimilar data points to be separated by a margin is encouraged by this.

### 4.3 Classification Model:
The feature vectors produced by the SNN's subnetwork is leveraged by the classification model to perform binary classification. It consists of a neural network architecture with layers such as dense layers and batch normalization. Pairs of images comprise the input to the classification model.

These pairs are processed by the model through the SNN's subnetwork to obtain their feature vectors.

To improve the model's generalization, the feature vectors are then normalized.

For predicting the probability of the input image pair belonging to class 1 (e.g., Alzheimer's disease), a final dense layer with a sigmoid activation function is responsible. The output of the sigmoid activation is a probability value between 0 and 1, where values closer to 1 indicate a positive classification (class 1), while values closer to 0 represent a negative classification (class 0).

### 4.4 Training and Classification:
Feeding labeled image pairs into the SNN and the classification model is involved in the training process. For effective classification the SNN's subnetwork learns to produce feature vectors.

The classification model is optimized to make accurate predictions based on the feature vectors from the SNN during training.

Using backpropagation and optimization techniques the models are fine-tuned to minimize the classification error.

The classification model can take pairs of new, unlabeled images and predict the probability of these images belonging to class 1, effectively classifying them as Alzheimer's disease or normal after training.

In summary, the SNN's ability to learn image representations that emphasize the similarity between pairs of images, especially for those of the same class is leveraged by this workflow. Binary predictions are then made by the classification model using these representations, allowing it to classify images as 0 (normal) or 1 (Alzheimer's disease) based on their similarity. Effective image classification is achieved by this approach while benefiting from the deep learning capabilities of neural networks.

## 5. Data 
### 5.1 Data Collection
The dataset used for this project was obtained from a particular URL: 

"<https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download>". The following steps were taken to acquire this dataset:

Using an HTTP GET request, the dataset was downloaded from the specified URL. In the utils.py file, the code responsible for downloading the file can be found. The dataset was saved locally in a specified file path after successful download. The downloaded content was written to this local file and was ensured by the code. The downloaded zip file was extracted to reveal its contents subsequently. Using the Python zipfile module, this extraction was performed. To store the extracted data, a designate directory was created. In case this directory did not exist, it was created by the code to make sure that the extraction process proceeded smoothly. The dataset files were then unzipped and were placed into the newly created directory. A list of the extracted files was generated (optional) to confirm that the extraction was successful for additional verification. These steps allowed us to obtain the dataset for further use in the project and constitute the data collection process.

![image](https://github.com/me50/s-prak/assets/116443738/8ecd2f11-3e9a-4500-902f-47465dd14777)

### 5.2 Data Preprocessing

### 5.3 Data Pairing
The preparation of data for training a Siamese neural network is streamlined by the load_siamese_data function that is often used for tasks like image similarity assessment. Initially, the function carries out image preprocessing. It encompasses reading and decoding images from JPEG format, resizing them to a standardized 128x128 pixel dimension, and scaling pixel values to a normalized range between 0 and 1. The consistency and readiness of the input data for neural network training is ensured by this.

The creation of image pairs is the core of the function and a fundamental aspect of Siamese network training. Pairs of images from the same class ("AD" or "NC") and pairs from different classes is constructed by it while assigning corresponding labels to indicate whether the pairs match (label 1) or not (label 0). The function deliberately randomizes the order of images within each class to eliminate any potential biases prior to pairing. It shuffles the dataset after pairing, ensuring that the order of pairs is randomized. Based on a user-defined ratio, the final dataset is split into training and validation sets and is then batched for efficient model training. This function simplifies the process of preparing data for Siamese neural network training, in summary, emphasizing the randomization of data before pairing and shuffling after pairing to promote unbiased and effective training.

## Results (Training and Testing)

## Python Packages
1.	Tensorflow
2.	Numpy
3.	Matplotlib
## References
1.	<https://medium.com/@rinkinag24/a-comprehensive-guide-to-siamese-neural-networks-3358658c0513>
2.	<https://www.baeldung.com/cs/siamese-networks>
3.	<https://builtin.com/machine-learning/vgg16>

Siamese: all loss 
Classifier: accuracy and the loss
 
Google Colab

![image](https://github.com/me50/s-prak/assets/116443738/c760824c-be8c-48f0-83e6-86c688037664)
![image](https://github.com/me50/s-prak/assets/116443738/0fd27eec-2eeb-46cd-a4aa-a4c9c899c4e9)
![image](https://github.com/me50/s-prak/assets/116443738/d896b303-0612-45ea-ad93-ac6f7c715eee)
![image](https://github.com/me50/s-prak/assets/116443738/de32015a-f031-4d48-bf56-abea541c63fb)
![image](https://github.com/me50/s-prak/assets/116443738/31378f42-0865-412b-88e8-50565f67cc27)
![image](https://github.com/me50/s-prak/assets/116443738/43b95742-2f6b-4154-962b-43c001713f8b)







