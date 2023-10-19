# Problem Description
The goal of this project is to diagnose Alzheimer's from a 2D MRI scan accurately using a visual transformer network. The data used was obtained from the "Alzheimer's Disease Neuroimaging Initiative" or ADNI dataset. Here are some samples:

![Alt text](readmeImages/altzSample.jpeg) ![Alt text](readmeImages/normSample.jpeg)
<pre>
Alzheimer's                         Normal Brain
</pre>

# The Visual Transformer

![Alt text](<readmeImages/Screenshot from 2023-10-13 22-47-10.png>)

The structure of the transformer was based off of the paper, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". The network has 3 distinct parts:

1): Embeddings: The images are cut up into patches, the patches are flattened, and each patch is fed into a single layer linear network to embed them. The idea here is that the network can learn a linear projection of the patches that is advantagious for classification. After the patches are projected, each embedded patch has a learned positional encoding added to it, this is to retain the positional information of the patches to aid with classification. Alongside these linear embeddings with positional encodings, there is an extra learned parameter, a "class embedding" added to position 0. This class embedding is what is used to classify the image after the transformer blocks.

2): Transformer Encoder: The structure of the encoder is on the right hand side of the image above. The "Norm" blocks are layernorm, standard in transformers. The skip connections are to stop the gradient signal from dying off and not effecting nodes at the start of a deep network. The mlp block at the end of the encoder exists so the model can learn to transform the output of the multi head attention into something maybe more useful for the next transformer encoder block. The multi-head (self) attention, however, is really the thing that makes this whole network work.

Multi-head self attention allows the network to find correlations between image patches that help it classify the image. It works by learning linear projections of the image patches for each head to convert them into keys, queries, and values, and then calculating attention on these, then concatenating them back together and then applying a learned linear projection again (as per the pytorch implementation). This layer learns the weights that allow it to relate key parts of the image with eachother.

The transformer encoder is repeated multiple times in the network structure usually.

3): MLP Head: This plays the same role as it would in a convolutional network, the transformer encoder blocks are essentially learned feature extraction, these dense layers take in the output of the transformer encoder and use it to classify the image as either a positive Alzeimer's case (outputs 1) or a negative (outputs 0).

# Training
The hyperparameters and model structure configuration are in train.py and dataset.py

The train and test splits were already done in the dataset downloaded off of the COMP3710 uq blackboard course website. The split was 9000 test points, and 21520 train points. This split was already a good balance between test and train sizes, so it wasnt changed

Each patient in the data set had 20 mri scans of their brain.

The validation set was produced by taking 216 patients, 108 with alzheimers and 108 without, (so 4320 samples or 20% of the train set). The validation split was done on a per patient basis to avoid data leakage from the train set into the validation set. If a patient appeared half in the validation set and half in the train set, the model might be just remembering the general brain shape or skull shape to get the alzheimer's classification right, giving a deceptively high validation accuracy, rather than looking in the correct places. A 20% validation set was chosen to give a good balance between decently estimating test performance and not cutting into the training set.

The images were cropped to be square, scaled to have its pixel values between 0 and 1. Data normalisation wasnt used since the pixel values were already scaled between 0 and 1.

The training data was augmented using pytorch's RandAugment function, applying 4 random augments to each train sample. This aided in overfitting, allowing the validation accuracy to improve for longer.

After training for 60 epochs the loss and accuracy curves were observed

![Alt text](readmeImages/FirstTrainAcc.png) ![Alt text](readmeImages/FirstTrainLoss.png)

At first the performance of the validation set is better than the train set, this is due to dropout and the data augmentation while training. Neuron outputs are being killed during training which hinders performance, then being all switched on during validation. In the case of data augmentation, the model is having a much harder time classifying on the train set since it is recieving images that are flipped, warped, cropped, etc. and then during validation the model recieves nice unaltered (besides the general scaling and square cropping) images.

Strangely enough the validation loss goes up, and is extremely noisy compared to the train loss. A possible explanation for the upward trend is that, since the model outputs a number between 0 and 1 with the prediction coming from rounding to the nearest integer, the wrong predictions are becoming more wrong. For example, by the end of training if the true label of a validation image is 1, the model might be outputting 0 (very wrong) rather than maybe earlier in training a 0.4 (still wrong but less so according to the loss function) for an a validation image with label 1.

As far as the noisyness goes, the validation data is very different from the train data as far as the model sees. All train data goes through 4 lots of random image augmentation, while the validation data goes though only scaling and square cropping. So while the model tries to minimise error on the train set, the decisions it makes might not directly carry over to the validation set, hence the noisyness. The train loss is the actual function being optimised, so it makes sense for that to decrease quite smoothly.

After this first 60 epochs, the model was trained for another 60 epochs, this time with a learning rate of 0.000001, 10 times less than before.


![Alt text](readmeImages/SecondTrainAcc.png) ![Alt text](readmeImages/SecondTrainLoss.png)

This time it can be seen that the model is starting to overfit, the validation accuracy stagnates while the train accuracy keeps increasing.

Notice at the end of the first accuracy graph and this accuracy graph, the accuracy stabilises a lot epoch to epoch. This indicates that further learning of the training data is no longer effecting validation performance, which is why it happens just before overfitting occurs, since overfitting is the problem of training accuracy gains adversly effecting validation accuracy. 

# Results

Testing the model after the first 60 epochs gave a test set accuracy of 67%. The difference between the validation set and test set accuracies are surprising. This is cant be attributed to data leakage between the train and validation set, since this was tested for extensively.

The only plausible reasons for this difference is natural variation in the test set compared to the validation set, or some very, very subtle bug in the code. This also doesn't seem to be entirely due to bias from hyperparameter tuning either since this trend of overfitting to the validation set was consistent throughout the training process.

# Requirements to Run and Libraries Used

The model was trained on a p100 gpu and took 2 hours to train for 60 epochs.

The torch and torchvision libraries were used. The data in its form used for training was obtained in its current form from the COMP3710 course blackboard website and is inaccessable to the general public. However the ADNI dataset is available to everyone, but is likely not in a form suitable for dataset.py. So if someone wanted to reproduce the results they would need to download the data themselves and make their own torch.utils.data.Dataset object with it.

dataset.py explains the file structure needed for it to properly initialise the dataset.

# License

This project is available under the MiT license
