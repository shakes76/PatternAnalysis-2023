# Siamese Network Classifier for Alzheimer's Disease Detection
## Description
This repository contains a Siamese Network-based classifier designed to identify Alzheimer's disease (AD) in brain data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The goal is to achieve an accuracy of approximately 0.8 on the test set by classifying brain images as either normal or AD.

## How It Works
This repository features a Triplet Siamese Network-based classifier for detecting Alzheimer's disease in brain data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The Triplet Siamese Network operates by taking three input images: an anchor image from a patient (either AD or normal), a positive image (another image from the same class as the anchor), and a negative image (from the opposite class). The network then computes feature embeddings for these three images. The primary objective is to minimize the distance between the anchor and positive images while maximizing the distance between the anchor and negative images. This ensures that the network learns to discriminate between AD and normal brain images effectively. The figure below illustrates the Triplet Siamese Network architecture:

![triplet_archi-1](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/2bc9f547-7923-4b4c-b5ba-4758dae840f9)


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

After this the TripletNet was trained for 100 epochs with the torch SGD optimiser with a learning rate set to 0.001.

The TripleNetClassifier was trained on the ADAM optimiser with a learning rate set to 0.001 for 50 epochs.

## Example Inputs, Outputs, and Plots
#### Inputs
The inputs to the Siamese Network classifier are pairs of brain images, one from an AD patient and one from a healthy individual. These image pairs are provided in the ADNI dataset.

#### Outputs
The output of the classifier is a binary classification result, indicating whether the input image pair is classified as normal or AD.

#### Plots
The following plots can be generated:

Training Loss and Accuracy: A plot of training loss and accuracy over epochs to visualize the training progress.
##### TripleNet Loss
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/38d0e0db-1b62-4517-a7da-a1b4883a88f3)

##### TripleClassifier Loss
![image](https://github.com/Kai-Barry/PatternAnalysis-2023/assets/88063818/18068dd2-98fd-4ade-b95e-45995860ee10)

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


## Data Splitting
The dataset is divided into three sets:

Training Set: Used to train the Siamese Network.
Test Set: Used to evaluate the model's performance and achieve the target accuracy of approximately 0.8.

For more detailed information, code implementation, and instructions on running the classifier, please refer to the accompanying Jupyter Notebook or Python script.
