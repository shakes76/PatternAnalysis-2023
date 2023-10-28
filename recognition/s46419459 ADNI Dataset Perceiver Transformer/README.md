# ADNI Classifying Perceiver Transformer Model
#### Deep Pais - id 46419459

## Introduction
The ADNI brain scan MRI Dataset splits brain scans (several layers of the brain) into cases of having Alzheimers, or Normal cognitive function. The goal of this report was to create an ML architecture that could classify the brain scans successfully into the correct class of diagnosis. For this report, a Perceiver transformer was create (task 6) to classify the scans. The perceiver architecture was originally created by Jaegle et al. (Source 1) to serve as a sort of Transformer architecture that does not assume the form of its input, but rather tries to find meaningful correlations within the inputs regardless of what they are. Additionally, the Perceiver architecture increases the scalability of inputs by forcing large high dimensional inputs through a latent bottleneck. 

The perceiver in this report was constructed using the pytorch machine learning library. The modules were guided by sources 1 and 2, and the architecture very closely follows that of the original perceiver conceived in the paper "Perceiver: General perception with iterative attention". Below is the general layout of the perceiver. For our test cases we used 6 blocks of attention->transformer modules:

![Perceiver Architecture](https://github.com/dcpais/PatternAnalysis-2023/blob/topic-recognition/recognition/s46419459%20ADNI%20Dataset%20Perceiver%20Transformer/figures/architecture.png?raw=true)

In the original perceiver architecture, the input image is originally passed through a positional embedding layer. This layer adds information as to where each pixel was in the image. Reason being is that the perceiver has no concept of position for the pixels, as data is passed through simultaneously. Thus positional embedding usually improves performance, as the cross attention modules can relate pixels to their neighbouring pixels. However, for this task there were some difficulties setting up positional embedding and thus it was substituted with a simple linear layer that reduced the image to the embedded dimensions as required for the cross attention blocks.

## Data preprocessing
The ADNI data is a large set of brain scan images. Each image is a slice of the brain, some having multiple layers (spread out over multiple images). The data was preprocessed as follows:
- Images were cropped to 240 x 240 size on the center
- Then they were converted to a pytorch tensor
- To flatten the channels of the image from RGB to 1 channel, they were converted to grayscale
- Lastly, the images were flattened into a singular byte array in preparation to enter the cross attention.

The data was pre split into a training and test. Training set comprised of ~65% of the dataset, whilst the test set held ~35%

## Reproducibility
1. To train the model, edit the "path" variable in train.py to lead to where you have stored the ADNI data. Path should end at '.../.../ADNI/'
2. Setup hyperparameters however you please (current hyperparameters were used for model training), as long as latent_dim << height*width of the images (240 * 240)
3. Run train.py to train the model and save it
4. To run some predictions, you can go into predict.py and edit path like in train.py. 
5. Then run the script and you will get one sample of each class and what the model predicted for it.

Below is an example prediction
![Example prediction](https://github.com/dcpais/PatternAnalysis-2023/blob/topic-recognition/recognition/s46419459%20ADNI%20Dataset%20Perceiver%20Transformer/figures/prediction.png?raw=true)

Note: Data was obtained from blackboard at https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI

## Results
Over 50 epochs, the following training results were achieved:
![Training results](https://github.com/dcpais/PatternAnalysis-2023/blob/topic-recognition/recognition/s46419459%20ADNI%20Dataset%20Perceiver%20Transformer/figures/lossaccplot.png?raw=true)

Additionally, test results after training were found to be:

![Test results](https://github.com/dcpais/PatternAnalysis-2023/blob/topic-recognition/recognition/s46419459%20ADNI%20Dataset%20Perceiver%20Transformer/figures/testacc.png?raw=true)

From the results, we see that the training accuracy bounces around 50% over all epochs. This is equivalent to the model randomly guessing the class of the image. 
Additionally, the loss of the model did not decrease over epochs as per usual, thus showing that over the epochs the model did not really improve or learn anything about the data.

There are several factors that could be held accountable. We could have set up the model to do more attention throughout each perceiver block to possibly boost performance, but seeing as over 50 epochs, there was no visible improvements there must have been some underlying error in the model that did not allow it to achieve any meaningful results. The lack of positional embedding in the model may have made it so that attention modules never actually yielded anything of significance, as positional information about pixels was never inferred in the training. If our hyperparameter selection could be a bit more expensive then we could have possibly started to see more results, but it is hard to know for sure.

### Dependencies
Python 3.11.3

pytorch 2.0.1+cu117

torchvision 0.15.2+cu117

Pillow 10.0.0

matplotlib 3.8.0

### References
1. https://arxiv.org/abs/2103.03206
2. https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb
