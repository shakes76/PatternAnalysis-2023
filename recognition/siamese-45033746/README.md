# Siamese network to classify Alzheimer's disease

### Algorithm Logic
The model comprises a triple siamese convolutional neural network and a binary classifier. The model is trained to 
classify 2D images slices of brain scans as Alzheimer's disease (AD) and normal cognitive (NC)

The siamese network consists of three convolutional neural networks, each with shared weights and structure. Of the CNN's, one is an anchor, one is a positive and one is a negative.
They take in a triplet of images, one of which is the anchor, one which is a positive match and one which is a negative match. For example, 
for Alzheimer's if the anchor is AD, then the positive image is also AD, and the negative is NC. 

During training, a contrastive loss function takes in the features of the network output and calculates loss to maximise 
the loss between the anchor and the negative, and minimise the loss between the anchor and the positive.

Once training is completed on the siamese network, we can use the siamese embeddings on the image input to another convolutional neural network,
this a binary classifier.

![Siamese triplet diagram](https://github.com/tweeeb/PatternAnalysis-2023/tree/topic-recognition/recognition/siamese-45033746/assets/triplet_siamese.jpg?raw=true)
### Data Pre-Processing
#### Transforms
Data is pre-proccessed with transforms such as random rotation, and horizontal flip to diversify data and prevent overfitting.

### Data partitions
Data is partitioned by patient to prevent data leakage. This is to prevent the model from seeing data from the same patient in the validation set or in the test set while being trained.

### Project Structure

### Dependencies

- matplotlib 3.7.2
- numpy 1.24.3
- Pillow 10.1.0
- torch 2.1.0

Install project requirements by running the following command
```
conda install --yes --file requirements.txt
```
### File Structure

This project uses the ADNI dataset for Alzheimer's disease. Please format the data under the directory AD_NC in 
this repository accordingly.

```
siamese-45033746
├── README.md
├── modules.py
├── dataset.py
├── predict.py
├── train.py
├── utils.py
├── .gitignore
└── AD_NC
    ├── test
    |    ├── AD
    |    |   └── 388206_78.jpeg
    |    |              .
    |    |              .
    |    |              .
    |    └── NC
    |    |   └── 1182968_88.jpeg
    |    |              .
    |    |              .
    |    |              .
    └── train
         ├── AD
         |   └── 218391_78.jpeg
         |              .
         |              .
         |              .
         └── NC
             └── 808819_88.jpeg
                        .
                        .
                        .
```