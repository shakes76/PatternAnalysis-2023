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
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

### Todo
- [x] Create my own training script that works on top of yolov7/train.py
- [x] Make a downsampled version of the dataset.
- [x] Manage a way of logging the runs to wandb
- [x] Zip and upload dataset to gdrive, and make functions for fetching the dataset from the cluster.
- [x] Make training scripts for use on the rangpur cluster.
- [ ] Manage to train on the downsampled dataset using the cluster. Need gpu version of CUDA for this to work.


### Installation

- Prerequisites: List any pre-required libraries or software. 
- Installation steps: 
  ```bash
  pip install -r requirements.txt
  ```

### Dataset
https://challenge.isic-archive.com/data/#2017
- Description of the dataset.
The original dataset consist of 2000 training images of skin lesions with corresponding labels and segmentation files for the specific part of the picture that is the lesion.
The validation set consist of 600 images with corresponding labels and the test set consists of 150 images with labels.


The different classes that are present in the dataset is: Melanoma, Nevi, and Seborrheic Keratosis. Where Nevi (Unknown) is the technical term for a benign skin lesion, often referred to as a mole.

- Preprocessing steps (if any).
Since the dataset comes with segmentation files preprocessing is needed to convert these lables into yolo bbox labels with the corresponding class.

This consist of mainly finding the max and min pixels of the segmentation area and fitting a bounding box to this area. The class is then determined by the label given in the csv file.

The dataset itself also needs to be refactored a bit to work with yolov7, therefore the structure is changed to the following:
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

### Contributing

- Details on how others can contribute to this project.
- Any specific guidelines for contributing.

### License

- Licensing information. For example:
  ```
  This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
  ```

### Acknowledgements

- Any credits or acknowledgments for third-party resources or researchers.

---

Remember, the above is just a basic skeleton. Depending on the project's complexity and breadth, you might want to add more sections or elaborate on the existing ones.