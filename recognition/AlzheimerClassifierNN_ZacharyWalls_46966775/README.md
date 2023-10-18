# AlzViT

This repository contains the code for training a custom made Visual-Transformer based model used to identify Alzheimer's disease in 2D sliced brain scans. The model was trained on the ADNI dataset which contains a number of sliced brain scan images separated into Normal (NC) and Alzheimer's (AD) classifications. The model is centered around the Vision Transformer (ViT) first introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" which works by dividing an image into fixed-size patches and leveraging self-attention mechanisms for feature extraction and pattern recognition.

## Dependencies

-   Python 3.x
-   PyTorch
-   torchvision
-   matplotlib
-   numpy

To install the required packages, run the following command:

```
pip install -r requirements.txt
```

## Dataset

The model is trained on the CelebA dataset, which can be downloaded automatically by the PyTorch `torchvision.datasets` module.

## Usage

1. Clone the repository:

```
git clone https://github.com/LeSnack/PatternAnalysis-2023-46966775.git
cd PatternAnalysis-2023-46966775/recognition/AlzheimerClassifierNN_ZacharyWalls_46966775
```

2. Train the model:

```
python train.py
```

This will train the AlzViT model on the ADNI dataset and save the trained model to `best_model_weights.pth` within the base AlzheimerClassifierNN_ZacharyWalls_46966775 folder.

3. Predict Disease:

```
python predict.py
```

## Model Architecture

## Results

## Acknowledgements

https://www.mdpi.com/2306-5354/10/9/1015
