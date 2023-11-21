# Improved Unet on ISIC2018 data

## The Problem
The ISIC challenge is an annual computer vision challenge that strides towards the classification of a large melanoma database. The challenge is split into levels with level 1 requiring segmentation masks to be made from images of skin legions. This project attempts to use the Improved Unet architecture to achieve a Dice similarity coefficient of 0.8 or greater.

## The Algorythm
![improved UNET Architecture](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/Improved_Unet_Architecture.png)

Unet was originally created in 2015 with the intended purpose of applying image segmentation to biomedical data. From there the Improved Unet was made in 2017, which was inspired by the original Unet but carefully optimized to maximise performance when applied to the segmentation of brain tumours for the BRaTs challenge. Whilst not the exact same thing, the increased performance of the model should translate to segmenting the ISIC 2018 dataset, which has images of skin legions.

The architecture of the Improved Unet model works by having multiple context layers that shrinks the data down, leaving behind only the most relevant features which is then passed through the localisation layers that reconstructs the features into an output. During all this, skip connections (dotted horizontal lines above) connect equivalent layers to be concatenated together so that the high resolutions details aren’t lost in the process. This is all eventually passed through a softmax layer to normalise the distribution of the outputs. This may not be necessary for this task though due to the binary nature of the outcome.

## How it works (setup and installs)
Create a new conda environment in miniconda3 with the needed libraries by using the following commands.
```bash
Conda create –name comp3710
Conda activate comp3710
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib
conda install tqdm
```

Python     = 3.9.16

pytorch    = 2.1.0

matplotlib = 3.7.2

tqdm       = 4.65.0

Once everything is installed, make sure to change the file paths in train.py (line 26-29) and predict.py (line 17-18) to where your data is installed.

In terms of preprocessing the data, it was already separated into training, validation and test before it was downloaded and left in those splits. In total there are 3694 images with 2594 used for training (70.2%), 100 used for validation (2.7%) and 1000 used for testing (27.1%).

## inputs and outputs
![gif of the 30 epoch progress](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/ezgif.com-gif-maker.gif)
![Epochs1](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/1-15.PNG)
![Epochs2](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/16-30%2Btime.PNG)
## Training and Validation Plots
![loss curve](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/LossCurve.png)
![dice curve](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/DiceCurve.png)

## Predictions
![results](https://github.com/valensmith/PatternAnalysis-2023/blob/topic-recognition/recognition/Project/images/result.PNG)

The output from predict.py shows that the model exceeds the goal Dice score of 0.8

## References
https://arxiv.org/abs/1802.10508v1

https://pytorch.org/docs/stable/index.html

https://www.geeksforgeeks.org/u-net-architecture-explained/
