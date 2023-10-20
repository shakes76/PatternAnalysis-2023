# Siamese network to classify Alzheimer's disease

### Algorithm Logic
The model comprises a triple siamese convolutional neural network and a binary classifier. The model is trained to 
classify 2D images slices of brain scans as Alzheimer's disease (AD) and normal cognitive (NC)

### Data Pre-Processing

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