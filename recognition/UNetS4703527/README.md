# UNet Segmention on the ISIC 2017/8 dataset

### Description: 
The task is to preform image segmentaion on the ISIC 2017/8 dataset. The implementation will employ a UNet convolutional network model for accurate segmentation. The UNet will take the given images and their coresponing Ground truth masks to which it will try to predict a mask. 

### Usage 

The UNet model can segment any type of image if it has enough traing data for both images and the ground truth masks.Therfore the model can be used on any dataset not just this one. In this senario we are using the UNet to segment skin lesions. This is benefical for medical imaging, containing the lesion to then take it further and detect irregularities in the lesion.

### ISIC 2017/8 dataset
The dataset consits of 2000 lesion images and 2000 ground truth masks. Preview of the first 10 images and their respective ground truth masks below.


<img width="440" alt="first masks" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/28c63790-50c7-40c7-94fd-443d288bfbb1">

### Architecture

![u-net-architecture](https://github.com/mraula/PatternAnalysis-2023/assets/96328895/14b488d2-e7bd-477e-a1b9-846d7e157e10)

#### Encoder Block

The encoder block has two 3x3 convolution layers with a ReLU then a max pooling. The encoder block is used to store features aswell as lower the dimentions of the image. The copy and crop is used as skip features in the decoder block from the stored features.

#### Bridging Block

The bridging block applies the two  two 3x3 convolution layers with a ReLU.

#### Decoder Block 

The decoder block has two 3x3 convolution layers with a ReLU then a up conv 2x2. The 4 encoder outputs are used as skip features in each decoder block. The Decoder tires to recunstruct based on the imput and skip features given.

### Results

#### Training 
<img width="549" alt="training results" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/4460d766-f2af-4a20-b876-93f993f8044f">

The data reached a val_dice_coef of > 0.8 after 13 epoch. As seen the val_dice_coef grew and val_loss droped with the num of epochs.

#### Evaluation


| Metric            | Score     |
|-------------------|-----------|
| Accuracy          | 0.95067   |
| Precision         | 0.90967   |
| Dice Coefficient  | 0.83335   |
#### Image results
<div style="display:flex;justify-content:space-between">
  <img width="400" alt="ISIC_0000001" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/a0bf3090-ef88-423e-ac65-eba66be9800c">
  <img width="400" alt="ISIC_0000003" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/2ff2892f-94a9-4acf-a1d2-6c48a1b1a293">
</div>

<div style="display:flex;justify-content:space-between">
  <img width="400" alt="ISIC_0000011" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/b0f62873-4900-4926-860e-b5ffd30be4ad">
  <img width="400" alt="ISIC_0000027" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/03e8575b-ea40-47a7-964d-24726347bbb9">
</div>

<div style="display:flex;justify-content:space-between">
  <img width="400" alt="ISIC_0000093" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/a77bb5bb-3a42-40b2-8724-f0fff51996dc">
  <img width="400" alt="ISIC_0000150" src="https://github.com/mraula/PatternAnalysis-2023/assets/96328895/84fa5471-f227-4b1d-a2a3-9d0034ba3865">
</div>

The results are shown as the Original image then the Groundtruth mask and finally the models Predicted mask.

### Repoducing Results

#### Train Model

To train the model run

```sh
python train --path "your-dataset-path"
```

Before training the model make sure you change the Dataset image mask path name in Datasets if you want to use a diffrent dataset. Make sure to also have a files folder in your durectort to store your data.csv and model.h5

#### Predict Model
To Predict the masks on the model run

```sh
pyton predict.py --path "your-data-set-path"
```
Make sure to have a results folder to save you predicted masks.

### Dependencies

This project requires the following dependencies:

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV
- Pandas
- Tqdm
### Refernces 

- https://medium.com/analytics-vidhya/image-classification-with-tf-keras-introductory-tutorial-7e0ebb73d044
- https://medium.com/geekculture/semantic-image-segmentation-using-unet-28dbc247d63e
- https://medium.com/@CereLabs/understanding-u-net-architecture-for-image-segmentation-74bef8caefee
- https://medium.com/mlearning-ai/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f#:~:text=Dice%20coefficient%20%3D%20F1%20score%3A%20a,recall%2C%20or%20vice%2Dversa.
- https://keviinkibe.medium.com/performing-image-segmentation-using-tensorflow-1c82608d2233
- https://arxiv.org/pdf/1505.04597.pdf
- https://medium.com/analytics-vidhya/image-input-pipeline-using-tensorflow-c9f729ead09f
- https://financial-engineering.medium.com/tensorflow-2-0-load-images-to-tensorflow-897b8b067fc2
- https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
- https://keras.io/examples/vision/oxford_pets_image_segmentation/

