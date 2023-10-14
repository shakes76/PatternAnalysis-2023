# AlzViT

This repository contains the code for training a custom made Visual-Transformer based model used to identify Alzheimer's disease in brain scans. The model was trained on the ADNI dataset for Alzheimer's disease with PyTorch utilised in the development and structuring of the model.

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
