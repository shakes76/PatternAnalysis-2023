## Lesion detection and classification using YOLOV7 on the ISIC2017 dataset

### Table of Contents
- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Todo](#todo)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Usage](#usage)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Results](#results)


### Installation
- Prerequisites: python=3.10.12 && cuda=11.7

**A GPU cluster is used for this project, more specifically rangpur @ UQ. Therefore a lot of the training and inference scripts are based on slurm jobs. If needed this can easily be converted to run locally.**

```
git clone git@github.com:larsmoan/PatternAnalysis-2023.git
git submodule init 
git submodule update
pip install -r requirements.txt
```

## Dataset
**Source**: [ISIC 2017 Dataset](https://challenge.isic-archive.com/data/#2017)

### Overview
Each image comes with corresponding label and segmentation file highlighting the lesion.
- **Training Set**: 
  - 2000 images.
  
- **Validation Set**: 
  - 600 images.
  
- **Test Set**: 
  - 150 images.

**Lesion Classes**:
- `Melanoma`
- `Seborrheic Keratosis`
- `Nevi / Uknown`: Technically known as a benign skin lesion. Commonly referred to as a mole.


### Preprocessing
Given that the dataset provides segmentation files, there's a need for preprocessing to convert these labels into YOLO bounding box labels. 

Steps include:
1. Identify the maximum and minimum coordinates within the segmentation area.
2. Fit a bounding box around this region.
3. Assign the class based on the label provided in the associated CSV file.

More information can be found in the file: [dataset_utils.py](./dataset_utils.py)

The dataset itself also needs to be refactored a bit to work with YOLOV7, therefore the structure is changed to the following:
```
dataset/
│
├── train/
│   ├── img_1.jpg
│   ├── ...
│   ├── img_n.jpg
│   ├── img_1.txt
│   ├── ...
│   └── img_n.txt
│
├── val/
│   ├── img_1.jpg
│   ├── ...
│   ├── img_n.jpg
│   ├── img_1.txt
│   ├── ...
│   └── img_n.txt
│
└── test/
    ├── img_1.jpg
    ├── ...
    ├── img_n.jpg
    ├── img_1.txt
    ├── ...
    └── img_n.txt
```

The prepocessed dataset can be downloaded from this link:
https://drive.google.com/uc?id=1YI3pwanX35i7NCIxKnfXBozXiyQZcGbL or from [dataset_utils.py](./dataset_utils.py)



### Usage
- Download the dataset and pretrained yolov7 weights:
  ```
  python dataset_utils.py
  ```
- Train the model:
  Using rangpur cluster:
  ```
  sbatch run_custom_train.sh
  ```
  Or using Google Colab:
  [isic_train.ipynb](./isic_train.ipynb)
- Run inference on testset:
  ```
  sbatch run_test.sh
  ```

### Model Architecture: open source [YOLOV7 Model](https://github.com/WongKinYiu/yolov7)

### Training
Training was mainly done on the rangpur cluster, using the P100 gpu and a batch size of 32

- Hyperparameters that was used is described in: [hyp.scratch.p6.yaml](./hyp.scratch.p6.yaml)

### Results
#### Confusion Matrix
<img src="results/yolov7_b32_p100/confusion_matrix.png" alt="Description" >


### F1 - curve
<img src="results/yolov7_b32_p100/F1_curve.png" alt="Description">

### Precision - Recall curve
<img src="results/yolov7_b32_p100/PR_curve.png" alt="Description">