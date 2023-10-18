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
https://drive.google.com/uc?id=1YI3pwanX35i7NCIxKnfXBozXiyQZcGbL

Which also happens to be the default argument to downloand_and_unzip() from the file dataset_utils.py


### Usage

- How to use the code/model.
- Example commands:
  ```bash
  python your_script_name.py --arg1 value1 --arg2 value2
  ```

### Model Architecture
The model and architecture used in this project is the open source yolov7 model:
https://github.com/WongKinYiu/yolov7


### Training
Training the yolov7 model on the dataset can be done in two different ways:
1. Using the isic_train.ipynb script
- This utilizes the free GPU provided by google-colab for training.

2. Using the rangpur cluster.
- Training can then be done using the run_custom_train.sh script. Here you can specify the number of epochs, batch size etc etc. Statistics from the training will be logged to the runs/ folder and wandb if that is enabled beforehand.

- Details about the training process.
- Hyperparameters used.
- Training time, hardware details (if relevant).

### Results

- Summarize the results obtained.
- Any metrics used (e.g., accuracy, F1-score).
- Visual results (e.g., plots, graphs) if applicable.