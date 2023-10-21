# Brain MRI super-resolution network 
## Introduction
This project implemented a a brain MRI super-resolution CNN network trained on the ADNI brain dataset. The trained model cnareconstruct a low resolution image into a high resolution version. The CNN consists of  three convolutional layers followed by an upscaling operation using depth-to-space transformation. The dataset used in training and testing is ADNI dataset, including 2D slices of MRI for both Alzheimer’s disease patients (AD) and health control (HC). For our purpose, dataset for AD and HC are combined together into one dataset (since our model do not deal with classification). The training and testing achives accracy of ? and ? respectively.
## Getting Started
### Install the required dependencies
pip install -r requirements.txt
### Loading dataset
The dataset used for training(and validation) and testing is loaded in dataset.py.The images are cropped into specified size. 20% of the training dataset is reserved for validation. Pixel values of traning and validation images are rescales to range of 0 to 1. A list of path for each test path is also created for later use.        
Then we produce paired high resolution correspong loss resolution images from the training and validation dataset. To get high resolution images,we convert images from the RGB color space to the YUV colour space and only keeps Y channel. To get low recolution version, we convert images from the RGB color space to the YUV colour space,only keeps Y channel and resize down sample them to specified size. Each pair is put into a tuple for traning purpose.
To run dataset.py, follow following steps:
1. Define directory containing training dataset(directly contains image files) at line ?? by changeing the variale data_dir
       

     
    
