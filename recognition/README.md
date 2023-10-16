# **Segment the ISIC 2017/8 data set with the Improved UNet** 
Released by the International Skin Imaging Collaboration (ISIC), the ISIC 2018 dataset was used in the annual Skin Lesion Analysis challenge.  
Below is a solution that segments the ISIC dataset using an improved U-Net architecture. The model aims to achieve a minimum DICE similarity coefficient of 0.8 on the test set for all labels.  

# **ISIC Dataset** 
We can download the test, training and validation set 2018 version from the ISIC website.  

### **Architecture** 
![Image](https://github.com/jyz523/PatternAnalysis-2023/assets/125327045/88cd0f74-a50f-4aaf-921f-76f108f943e2)


### **Training DSC and Loss Plots over Epochs:** 
![Image](https://github.com/jyz523/PatternAnalysis-2023/assets/125327045/6ad6ed69-5483-4f55-89ce-1cad501deb80)


Average Dice Similarity Coefficient: 0.7197

## **Dependencies** 
torch 2.1.0  
matplotlib 3.8.0  
numpy 1.26.0  
torchvision 0.16.0  
Python 3.11

## **Reference** 
[1] K. He, G. Gkioxari, P. Dollár, and R. Girshick, “Mask R-CNN,” in 2017 IEEE International Conference on
Computer Vision (ICCV), Oct. 2017, pp. 2980–2988.
