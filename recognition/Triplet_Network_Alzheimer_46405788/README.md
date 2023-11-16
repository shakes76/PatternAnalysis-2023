# Siamese Network Classifier for Alzheimer's Disease Detection
## Description
This repository contains a Siamese Network-based classifier designed to identify MRI images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset as either Alzheimer's disease (AD) or Cognitive Normal (NC). The goal is to achieve an accuracy of 0.8 on the test set.

## How It Works
This repository features a Triplet Siamese Network-based classifier for detecting Alzheimer's disease in brain data from the ADNI dataset. The Triplet Siamese Network operates by taking three input images, an anchor image (which would be either AD or NC), a positive image (another image thats the same class as the anchor), and a negative image (an image that is the opposite class of the anchor). The network then computes feature embeddings for these three images. The primary objective is to minimize the distance between the anchor and positive images while maximizing the distance between the anchor and negative images. This ensures that the network learns to discriminate between AD and NC brain images effectively. Once the network has learnt to discriminate between AD and NC brain images effectivley, a neural network will learn to classify an image based off of the embeddings as either AD or NC. The figure below illustrates the Triplet Siamese Network architecture:

![triplet_archi-1](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/2bc9f547-7923-4b4c-b5ba-4758dae840f9)
**Source:** https://pyimagesearch.com/2023/03/06/triplet-loss-with-keras-and-tensorflow/
### Triplet Loss
As stated earlier, the primary objective is to minimize the distance between the anchor and positive images while maximizing the distance between the anchor and negative images. This is done through triplet loss which follows these steps:
1. Calculate the Squared Euclidean distance between the anchor and the positive
2. Calculate the Squared Euclidean distance between the anchor and the negative
3. Calculate the difference between the squared distances
4. Add the margin to the differnce
5. Apply a ReLU to ensure the loss is positive.
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/ffbb0047-deac-4760-a5a2-6930efd2e717)
**Source:** https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905
## Architecture
### Triplet Siamese Network Architecture
The triplet siamese network uses ResNet 18 as its backbone for creating the embeddings for the ADNI dataset. Residual Network (ResNet), is a type of convolutional neaural network commonly used for vision and image recognition. The key difference from a regular convolutional neaural network, is the use of residual blocks. A common issue with networks with high number of layers, is the vanishing gradient problem.. This is when the gradient becomes too small during backpropagation making it difficult to train networks with large number of layers. The resnet introduces the skip connection which bypasses certain layers which addresses the vanishing gradient problem. As can be seen in the figure below, ResNet 18 has 18 convolutional layers. The triplet siamese network, will take a 1 channel ADNI image (i.e grayscale) and output the image in a 128 dimension embeded space.
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/a1888ac3-d9e2-4558-b984-284374083703) <br>
**Source:** https://www.researchgate.net/figure/Original-ResNet-18-Architecture_fig1_336642248

### Triplet Classifier Architecture
The triplet classifier uses a basic neural network to classify between the AD and NC. The first fully connected layer has 128 input neurons and 256 output neurons followed by a `ReLu`. The second linear layer takes the 256 outputs from the previous layer and outputs 256 neurons to another `Relu`. The final layer takes the previous 256 outputs and outputs to 2 units. These 2 units represent the two classes, where the larger value between the two, is the predicted class. 


## Dependencies
To run this code, you'll need the following dependencies:

* Python (version 3.8)
* PyTorch (version 2.0.1)
* NumPy (version 1.24.3)
* Matplotlib (version 3.7.1)

## Reproducibility
To ensure reproducibility of results, it's recommended to create a virtual environment and specify the exact versions of the dependencies used.
The TripletNet was trained for 35 epochs with the torch SGD optimiser with a piecewise learning schedular represented below, where learning_rate is set to 0.1:

CyclicLR Scheduler (`sched_linear_1`):
* `optimizer`: The optimizer for which the learning rate schedule is applied.
* `base_lr`: The minimum learning rate during the cycle. It's set to 0.005.
* `max_lr`: The maximum learning rate during the cycle. It's set to learning_rate.
* `step_size_down`: The number of iterations to reach the minimum learning rate after each cycle's peak. It's set to 15.
* `mode`: The mode of the cyclic learning rate, which can be 'triangular' or 'triangular2'. In your case, it's 'triangular'.
* `verbose`: A boolean flag to indicate whether you want verbose output. It's set to False, so you won't get verbose information.

LinearLR Scheduler (`sched_linear_3`):
* `optimizer`: The optimizer for which the learning rate schedule is applied.
* `start_factor`: The initial learning rate factor. It's set to 0.005/learning_rate.
* `end_factor`: The final learning rate factor. It's set to 0.005/5.
* `verbose`: A boolean flag to indicate whether you want verbose output. It's set to False.
  
SequentialLR Scheduler (`scheduler`):
* `optimizer`: The optimizer for which the learning rate schedule is applied.
* `schedulers`: A list of learning rate schedulers to apply sequentially. In your case, you're applying `sched_linear_1` followed by `sched_linear_3`.
* `milestones`: A list of iteration milestones that trigger the transition between different learning rate schedulers. The transition between `sched_linear_1` and `sched_linear_3` occurs at iteration 30.

After this the TripletNet was trained for 100 epochs with the torch `SGD` optimiser with a learning rate set to 0.001.

The TripleNetClassifier was trained on the ADAM optimiser with a learning rate set to 0.001 for 50 epochs.

## Example Inputs, Outputs, and Plots
#### Inputs
The inputs to the Siamese Network classifier are pairs of brain images, one from an AD patient and one from a healthy individual. These image pairs are provided in the ADNI dataset.

#### Outputs
The output of the classifier is a binary classification result, indicating whether the input image pair is classified as NC or AD (0, 1).

![report](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/e21ab09e-bc18-4bdc-9c4d-a4d3954038bf)

#### Plots
The following plots can be generated:

Training Loss and Accuracy: A plot of training loss and accuracy over epochs to visualize the training progress.
##### TripleNet Loss 35 epochs with the `scheduler`
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/38d0e0db-1b62-4517-a7da-a1b4883a88f3)
##### TripleNet Loss 100 epochs with `SGD` optimiser with a lr = 0.001
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/0c906e37-583e-4d50-a830-7585b603891e)


##### TripleClassifier Loss
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/18068dd2-98fd-4ade-b95e-45995860ee10)

## Results
After training the TripletNet, on the ADNI dataset for the 135 total epochs the tripleNet has the following accuracies:

**Triplet Network accuracy**
* Test Accuracy: 80%

After training the TripletClassifier, with the previously seen 80% accurate model creating the emmbeddings, the TripletClassifier has the following accuracies:

**Triplet Classifier accuracy**
* Test Accuracy: 77.7%
  * AD Accuracy: 73.7%
  * NC Accuracy: 81.6%

#### Batch Example
This was an example of 40 randomly selected images from the test split with its corresponding classifcations and the tripleClssifer predictions:
* y_test_batch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
* y_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]

#### Test Data TSNE Embedding
The embeddings can be graphed when using a dimensionality reduction alorithm. For this example, we are using TSNE to represent the original 128 dimension embedding to a 2 dimensional embedding. Due to the large dimensionality reduction, we are loosing a fair bit of information, thus the data seems to have a large overlap. Although some overlap, the general pattern can be seen where the AD in red stay on the left side while the NC in blue stay on the right side.
![tsne](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/09877421-962c-48af-b839-3934fecc3d51)

## Pre-processing 

#### Train Transform:

- **Resize:** The images are resized to a fixed size of 100x100 pixels using `transforms.Resize((100, 100))`.

- **Random Horizontal Flip:** This transformation randomly flips images horizontally, which can help augment the training data and improve the model's ability to handle mirrored or flipped images. `transforms.RandomHorizontalFlip()` is used for this purpose.

- **To Tensor:** The `transforms.ToTensor()` step converts the image into a PyTorch tensor. This is a necessary step because PyTorch works with tensors as input data.

- **Normalize:** The `transforms.Normalize` step standardizes the image data. It subtracts the mean values `[0.485, 0.456, 0.406]` from the image and then divides by the standard deviation values `[0.229, 0.224, 0.225]`. This normalization process helps ensure that the pixel values are in a suitable range for model training.

- **Random Crop:** A random crop of size 100x100 pixels is applied with padding of 4 pixels using the 'reflect' padding mode. This helps create data augmentation by selecting different parts of the image during training. `transforms.RandomCrop(100, padding=4, padding_mode='reflect')` is used for this.

- **Grayscale:** Finally, the image is converted to grayscale using `transforms.Grayscale()`. This step converts the image to a single-channel grayscale format, which can be useful for some specific applications.

#### Test Transform:

- **Resize:** The images are resized to a fixed size of 100x100 pixels using `transforms.Resize((100, 100))`.

- **To Tensor:** The `transforms.ToTensor()` step converts the image into a PyTorch tensor.

- **Normalize:** Similar to the train transform, the `transforms.Normalize` step is used to standardize the image data by subtracting the mean values `[0.485, 0.456, 0.406]` and dividing by the standard deviation values `[0.229, 0.224, 0.225]`.

- **Grayscale:** Finally, the image is converted to grayscale using `transforms.Grayscale()`, just as in the train transform.

## Data Directory Structure

To use the data with the default values, make sure to organize your data as shown below:

![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/39dbd207-7a21-494c-889e-a4acd378a689)



## Data Splitting
The dataset is divided into two sets:

Training Set: Used to train the Siamese Network.
Test Set: Used to evaluate the model's performance and achieve the target accuracy of approximately 0.8.

For more detailed information, code implementation, and instructions on running the classifier, please refer to the accompanying Jupyter Notebook or Python script.
