# Improved Unet on ISIC2018 data

## The Problem
The ISIC challenge is an annual computer vision challenge that strides towards the classification of a large melanoma database. The challenge is split into levels with level 1 requiring segmentation masks to be made from images of skin legions. This project attempts to use the Improved Unet architecture to achieve a Dice similarity coefficient of 0.8 or greater.

## The Algorythm
![improved UNET Architecture](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/Improved_Unet_Architecture.png)

Unet was originally created in 2015 with the intended purpose of applying image segmentation to biomedical data. From there the Improved Unet was made in 2017, which was inspired by the original Unet but carefully optimized to maximise performance when applied to the segmentation of brain tumours for the BRaTs challenge. Whilst not the exact same thing, the increased performance of the model should translate to segmenting the ISIC 2018 dataset, which has images of skin legions.

The architecture of the Improved Unet model works by having multiple context layers that shrinks the data down, leaving behind only the most relevant features which is then passed through the localisation layers that reconstructs the features into an output. During all this, skip connections (dotted horizontal lines above) connect equivalent layers to be concatenated together so that the high resolutions details aren’t lost in the process. This is all eventually passed through a softmax layer to normalise the distribution of the outputs. This may not be necessary for this task though due to the binary nature of the outcome.


## inputs and outputs
![gif of the 30 epoch progress](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/ezgif.com-gif-maker.gif)

## Training and Validation Plots
![loss curve](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/LossCurve.png)
![dice curve](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/DiceCurve.png)
