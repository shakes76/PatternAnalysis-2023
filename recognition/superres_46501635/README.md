ADNI
│
└───AD_NC
    │
    ├───test
    │   ├───AD
    │   │   └───*.jpeg (multiple jpeg images)
    │   │
    │   └───NC
    │       └───*.jpeg (multiple jpeg images)
    │
    └───train
        ├───AD
        │   └───*.jpeg (multiple jpeg images)
        │
        └───NC
            └───*.jpeg (multiple jpeg images)
---

# ESPCN Super-Resolution Model

## Overview

This project implements the Efficient Sub-Pixel Convolutional Neural Network (ESPCN) for image super-resolution. The ESPCN model is designed to perform super-resolution in the low-resolution (LR) space, which reduces computational complexity. The model replaces the handcrafted bicubic filter in the super-resolution pipeline with learned upscaling filters, offering improved performance.

## Key Features

- **Feature Extraction in LR Space:** ESPCN extracts feature maps in the LR space, allowing for reduced computational complexity.
- **Sub-Pixel Convolution:** The model uses a sub-pixel convolution layer to upscale the final LR feature maps to high-resolution (HR) output.
- **Custom Dataset Handling:** The dataset is organized into training and testing sets, with categories for AD and NC images.
- **Visualization:** The training script provides visual feedback on the model's performance every few epochs.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib

### File Structure

```
.
├── modules.py          # Contains the ESPCN model definition
├── dataset.py          # Contains the custom dataset class and data loaders
├── train.py            # Contains the training loop and visualization
└── predict.py          # Contains the prediction script to visualize model outputs on test data
```

### Usage

1. **Training the Model:**
    ```bash
    python train.py
    ```
    This will train the ESPCN model on the provided dataset and save the trained model weights.

2. **Predicting with the Model:**
    ```bash
    python predict.py
    ```
    This will load a trained ESPCN model and make predictions on a sample from the test dataset. The results will be visualized.

### Model Details

- **Loss Function:** Mean Squared Error (MSE) loss is used as it ensures that the reconstructed high-resolution image is pixel-wise similar to the original.
- **Optimizer:** Adam optimizer is employed due to its faster convergence and better performance in deep learning tasks, including super-resolution.
- **Learning Rate Scheduler:** `ReduceLROnPlateau` is used to adjust the learning rate based on the model's performance.
- **Performance Metric:** Peak Signal-to-Noise Ratio (PSNR) is used to measure the quality of the reconstructed image compared to the original.

### Notes

- Ensure you adjust file paths in the scripts if you're using a different directory structure.
- The model is trained on grayscale images. Ensure your dataset consists of grayscale images or modify the model to handle RGB images.

## Acknowledgements

This project is based on the principles of the ESPCN model for super-resolution. The implementation is inspired by various research papers and PyTorch tutorials.

---

You can save the above content in a `README.md` file at the root of your project directory. Adjustments can be made as per your specific requirements or additional details you'd like to include.
