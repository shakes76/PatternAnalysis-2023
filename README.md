# Segment the ISIC 2017/8 data set with the Improved UNet
Released by the International Skin Imaging Collaboration (ISIC), the ISIC 2018 dataset was used in the annual Skin Lesion Analysis challenge.  
Below is a solution that segments the ISIC dataset using an improved U-Net architecture. The model aims to achieve a minimum DICE similarity coefficient of 0.8 on the test set for all labels.  

### ISIC Dataset
We can download the test, training and validation set 2018 version from the ISIC website.  

### Architecture
![Image](https://github.com/jyz523/PatternAnalysis-2023/assets/125327045/88cd0f74-a50f-4aaf-921f-76f108f943e2)
Figure 1(above), Our design was inspired by the U-Net. Top level data is gathered by the context left pathway and then precisely localized by the localization right pathway.  Through deep supervision, we introduce gradient signals deeply into the network.

### Data Preparation  
- Feature extraction and normalization to prepare raw data for the model.  
- Data augmentation techniques to enhance model robustness and generalization.  
- Segregation of data into distinct training and validation sets for effective model evaluation.  
 
### Model Building  
- Construction of model architecture using appropriate layers (e.g., convolutional, recurrent, etc.), considering the specific requirements of the task.  
- Integration of activation functions to introduce non-linearities essential for learning complex patterns.  
- Implementation of regularization methods to prevent overfitting and improve model generalization.  

### Training Procedure  
- Selection and application of a relevant loss function to assess model predictions during training.  
- Utilization of optimizers to adjust model parameters based on the computed loss, enhancing model performance.  
- Systematic training process over multiple iterations (epochs), using mini-batches of data for efficient learning.  

### Prediction and Evaluation  
- Execution of model inference to generate predictions on new, unseen data.  
- Assessment of model performance post-training using task-relevant metrics to determine accuracy and reliability.  
- Iterative model tuning and hyperparameter adjustments based on evaluation feedback to optimize results.  
- Each of these points represents a critical stage in the machine learning model's lifecycle, from initial data handling to the model building, followed by the training phase, and culminating in evaluation and optimization based on performance feedback.  

### Performance Metric
The Dice Similarity Coefficient (DSC) was used to evaluate the model. DSC is a statistic used to compare the similarity of two sets, most commonly use when comparing two images in biomedical image processing.  

### Training DSC and Validation Plots:
![Image](https://github.com/jyz523/PatternAnalysis-2023/assets/125327045/80bd011b-f776-4c41-a9ae-3dd136d01a19)  
(Only can run 15 Epoch due to hardware)  
Training Dice Similarity Coefficient: 0.746   
Validation Dice Similarity Coefficient: 0.725
-> Average DC in testing dataset: 0.85765  
### Visual PLot:  
<img width="332" alt="result" src="https://github.com/jyz523/PatternAnalysis-2023/assets/125327045/50862ba9-3cb9-44da-b4ae-6dfb76239ad2">


### Dependencies 
torch 2.1.0  
matplotlib 3.8.0  
numpy 1.26.0  
torchvision 0.16.0  
Python 3.11  

## **Reference** 
[1] K. He, G. Gkioxari, P. Dollár, and R. Girshick, “Mask R-CNN,” in 2017 IEEE International Conference on
Computer Vision (ICCV), Oct. 2017, pp. 2980–2988.  
[2] Fabian Isensee, Philipp Kickingereder, Wolfgang Wick, Martin Bendszus, Klaus H. Maier-Hein ,Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge  
