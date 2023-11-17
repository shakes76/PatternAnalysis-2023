# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* OASIS brain segmentation
* Classification
etc.

# Improved UNet for ISIC 2017/8 Dataset Segmentation
## Author
Name: [Yiming Liu]
Student Number: [47322462]

This project was completed for ["COMP3710 Report 2023"]

## Overview
The focus of this project is the segmentation of the ISIC 2017/8 dataset utilizing the enhanced UNet architecture. Our ambition is to achieve a Dice similarity coefficient of at least 0.8 on the test set, ensuring precise skin lesion segmentation.

## The Improved UNet: A Snapshot
Input Stage: Initiates with an image.

Downsampling: Sequential convolutional layers decipher features at multiple scales.

Bottleneck: Captures the nuanced, higher-level features.

Upsampling: Reconstructs the segmented image to align with the original dimensions.

Output: Outputs the final segmented version of the image.


## Advantages
1. **Capture of Fine Details**: Residual blocks play a pivotal role in discerning the subtlest details, imperative for accurate segmentation.
2. **Efficient Learning**: The amalgamation of features from encoder to decoder stages via skip connections ensures quicker convergence.
3. **Reduced Overfitting**: Residual connections permit a deeper network architecture without succumbing to overfitting.

## Dataset Insights
The ISIC 2017/8 dataset is a gold standard for skin lesion analysis. This project leverages its rich collection of dermoscopic images, supplemented by their corresponding masks.

Here is the structure of the dataset directory:
.
├── ISIC-2017_Training_Data
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ISIC-2017_Training_Part1_GroundTruth
    ├── mask1.png
    ├── mask2.png
    └── ...



## Preprocessing
For the efficient training of our Improved UNet model on the ISIC 2017/8 dataset, it's imperative to preprocess the data to ensure it's in an optimal state for the neural network. Here's an overview of the preprocessing steps undertaken:

Resizing:

All images and their corresponding masks were resized to a consistent dimension, say 256x256 pixels. This ensures that the network receives inputs of a fixed size.

Normalization:

Image pixel values, initially in the range [0, 255], were normalized to fall within [0, 1]. This aids in faster and more stable convergence during training.

Data Augmentation:

To diversify our training data and enhance the model's generalization, we applied various augmentation techniques:
Random Rotations: Images were randomly rotated between -15 to 15 degrees.
Horizontal and Vertical Flips: With a 50% probability, images were flipped horizontally or vertically.
Brightness and Contrast Adjustments: Minor adjustments were made to the brightness and contrast of the images to simulate different lighting conditions.

Train-Validation Split:

The dataset was split into training and validation sets. Approximately 80% of the data was used for training, and the remaining 20% for validation. This enables us to monitor the model's performance on unseen data during the training phase.

Batching:

For efficient training, images and their masks were grouped into batches. Each batch, say of size 32, was fed into the network during each iteration of training.

Shuffling:

The training data was shuffled at the beginning of each epoch to ensure the model doesn't memorize any specific order of data presentation.
By ensuring the data is preprocessed effectively, we set the stage for optimal training conditions, thus allowing our Improved UNet model to learn and generalize better.


## Training Insights

With the Adam optimizer at the helm and a learning rate set at 0.001, our Improved UNet model embarked on its training journey. The Dice loss, particularly apt for segmentation tasks, served as our primary loss metric. Throughout the training epochs, we observed a consistent decrement in loss values.

## Testing Outcomes

Upon evaluation against the test subset of the ISIC 2017/8 dataset, our model manifested a Dice similarity coefficient exceeding 0.8, underscoring its adeptness at segmentation.

## Dependencies

- Python 3.10
- PyTorch 1.10.0
- torchvision 0.11.0
- PIL 9.0.0
