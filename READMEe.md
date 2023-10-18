Improved UNet for ISIC 2017/8 Dataset Segmentation
Author
Name: [Yiming Liu]
Student Number: [47322462]

This project was completed for ["COMP3710 Report 2023"]

Description
This project aims to segment the ISIC 2017/8 dataset using the Improved UNet architecture. 
The primary goal is to achieve a minimum Dice similarity coefficient of 0.8 on the test set, ensuring high-quality segmentation of skin lesions.


Improved UNet
How It Works
Input Stage: The architecture begins with an image input.
Downsampling: The image is passed through a series of convolutional layers to capture features at various scales.
Bottleneck: This section captures higher-level features.
Upsampling: The architecture then reconstructs the image from the features, ensuring that the segmented image matches the original size.
Output: The final layer produces the segmented image.

Advantages
1. **Capture of Fine Details**: The residual blocks aid in capturing finer details which are crucial for accurate segmentation.
2. **Efficient Learning**: Skip connections ensure that features from encoder stages are combined with the decoder stages, which helps in faster convergence.
3. **Reduced Overfitting**: Due to the integration of residual connections, the network can be deeper without the fear of overfitting.

Dataset
The dataset used is the ISIC 2017/8, which is designed for skin lesion analysis.
The project utilizes the ISIC 2017/8 dataset for skin lesion segmentation. The dataset comprises dermoscopic images of various skin lesions, providing both the images and their corresponding masks.

Here is the structure of the dataset directory:
.
├── ISIC-2017_Training_Data
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
└── ISIC-2017_Training_Part1_GroundTruth
├── mask1.png
├── mask2.png
└── ...


Preprocessing

Training, Validation, and Testing

## Training

The Improved UNet model was trained using the Adam optimizer with a learning rate of 0.001. 
The primary loss function used was the Dice loss, which is apt for segmentation tasks. 
The model underwent rigorous training, and the training process exhibited a consistent reduction in loss over epochs.

## Testing

The model's performance was evaluated on the test set of the ISIC 2017/8 dataset. 
It achieved a Dice similarity coefficient above 0.8, indicating precise segmentation capabilities.

## Dependencies

- Python 3.10
- PyTorch 1.10.0
- torchvision 0.11.0
- PIL 9.0.0

