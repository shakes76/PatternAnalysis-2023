# Segment the ISIC 2017/8 data set with the Improved UNet 
Released by the International Skin Imaging Collaboration (ISIC), the ISIC 2018 dataset was used in the annual Skin Lesion Analysis challenge.  
Below is a solution that segments the ISIC dataset using an improved U-Net architecture. The model aims to achieve a minimum DICE similarity coefficient of 0.8 on the test set for all labels.  

# ISIC Dataset  
We can download the test, training and validation set 2018 version from the ISIC website.  

#Training DSC and Loss Plots over Epochs:  
![Image]([https://github.com/jyz523/PatternAnalysis-2023/assets/125327045/6ad6ed69-5483-4f55-89ce-1cad501deb80](https://github.com/jyz523/PatternAnalysis-2023/blob/topic-recognition/recognition/Plot/Epoch%2015%2073%25.png)https://github.com/jyz523/PatternAnalysis-2023/blob/topic-recognition/recognition/Plot/Epoch%2015%2073%25.png)

Average Dice Similarity Coefficient: 0.7197

* Object detection

