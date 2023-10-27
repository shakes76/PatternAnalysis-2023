## COMP_3710_Report

## Medical condition, Extended to 27 OCT

## This report is focused on the first task (a)
   Segment the ISIC data set with the Improved UNet
   with all labels having a minimum Dice similarity coefficient of 0.8 on the test set.

## Background
The structure of Improved UNet is based on the paper. 
"Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge" 
https://arxiv.org/abs/1802.10508v1


## Model
   U-Net is a convolutional neural network architecture primarily used for biomedical image segmentation. 
   Its U-shaped structure consists of a contracting path, which captures context, and an expansive path,
   which enables precise localization. Through skip connections, features from the contracting path are concatenated
   with the expansive path, enhancing localization capabilities.

   ![Unet](./additional_images/unet.png)

   The activations in the context pathway are computed by context modules. 
   Each context module is in fact a pre-activation residual block with two 3x3x3 convolutional 
   layers and a dropout layer (p_drop = 0.3) in between. Context modules are connected 
   by 3x3x3 convolutions with input stride 2 to reduce the resolution of the feature maps and allow 
   for more features while descending the aggregation pathway.

## Dataset
   In this report, the ISIC 2018 dataset will be used. 
   The ISIC 2018 dataset is a publicly available dataset for skin lesion image segmentation,
   provided by the International Skin Imaging Collaboration (ISIC). Given that the real-world
   images in the dataset come in different sizes, they are uniformly resized to a 128x128 dimension.
   These images use RGB with 3 color channels for input. The label data, which indicates where the lesions are,
   is treated in the same way as the real data. However, these labels are input as grayscale images with a single channel,
   making them simpler and more focused on the lesion's location and shape.


## Data_Loader
Class for getting data as a Dict

    Args: 
    images_dir = path of input images
    labels_dir = path of labeled images 
    transformI = Input Images transformation 
    transformM = Input Labels transformation 
   
## Dice_score
measures the similarity between two sets. 
Specifically in medical imaging, it's used to quantify the overlap 
between predicted and ground truth binary segmentations.
Values range between 0 (no overlap) to 1 (perfect overlap). 
It's a common metric for evaluating the accuracy of image segmentation models, 
highlighting the spatial overlap accuracy between prediction and truth.

## Losses
Quantifies how well a model's predictions match the actual data. 
In machine learning, it measures the difference between predicted and true values. 

    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch

## Environment
    python 3.9    
    pytorch=2.0.1
    numpy==1.23.5
    torchvision=0.15.2
    matplotlib==3.7.2
      Pillow==9.3.0
      tqdm==4.64.1
      wandb==0.13.5

## Training loss vs Epoches

   ![train_vs_Epoches](./additional_images/Train_loss_vs_Epoches.png)


## Valid loss vs Epoches

   ![valid_vs_epoches](./additional_images/valid_loss_vs_epoches.png)


## Predict
   The input is 

   ![Input](./additional_images/ISIC_0000003.jpg)


   The output is  

   ![Output](./additional_images/ISIC_0000003_out.jpg)


    
## Results
   The final model get a Mean Dice Score : 0.9940897984524689.


   











