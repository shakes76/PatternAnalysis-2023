# Classify Alzheimer’s disease of the ADNI brain data using perceiver transformer
### By Rohan Kollambalath

## Introduction

In this project I use the perceiver transformer to classify if MRI brain scans have Alzheimers disease with the ADNI dataset. The goal of the project was to achieve above 80% accuracy in the test set and produce a model that does not make too many assumtions about the input data.

The Perciever Transformer is an iteration of the transformer model released by Google Deep-Mind to solve its quadratic bottleneck and modalility limitations. [1]: [Perceiver: General Perception with Iterative Attention] (https://arxiv.org/pdf/2103.03206.pdf)

it solves the quadratic bottleneck by projecting the input information into a lower dimention latent array through cross attention layers similar to those in traditional transformers. The array is then processed through multiple self attention layers to extract features. This process also allows the perciever to make less structural assumptions about the input data allowing for many different modes of data as inputs including text, audio, video etc. 

![PercieverTransformer](../plots/perciever_transformer.jpg)


## Set-Up and Dependencies
This project uses pytorch to build the perceiver in the same regard as the original paper. I used object oriented programming for all the components of the model and dataloader. 
Architectural decisions for this model were influenced by "The Annotated Perciever" article [2]: [The Annotated Perceiver](https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb) as well as the original paper [1]: [Perceiver: General perception with iterative attention. In International conference on machine learning](https://arxiv.org/abs/2103.03206)

Dependencies:
 - Python 3.9.2
 - PyTorch 2.0.1
 - Matplotlib 3.7.1

## Implementation and Reproducability
This model forllows the architecture proposed in the paper. It consists of 3 perciever blocks each consisting of a singular head cross attention module and 4 self attention modules with 8 heads each. A batch size of 32 was used with BCELoss criteron as this is a binary classification problem. The adam optimiser was used instead of the LAMB optimiser proposed in the model.


HyperParameters:
 -  LATENT_DIM = 32
 -  LATENT_EMB = 128
 -  latent_layers = 4
 -  latent_heads = 8
 -  classifier_out = 16
 -  batch_size = 32
 -  epochs = 60
 -  depth=3
 -  learning_rate = 0.005

To train your model run the file train.py with your desired hyperparameters or instantiate a new model with
 -  ADNI_Transformer(depth, LATENT_DIM, LATENT_EMB, latent_layers, latent_heads, classifier_out, batch_size)


### Data Processing
The ADNI dataset has already been split into adequate test and train folders by the teaching team. The datasets have both AD (has alzheimers) and ND (no alzheimers) labels. I used pytorch to load the data into a dataloeader with a batch size of 32. Transformations were applied to the data including cropping images from 240x256 to 240x240 as well as converting to black and white. The data was also normalised to a mean of 0 and an STD of 1.

The data split is 80% train and 20% test. Validation sets were not used in training this model as the dataset is already small for a transformer and data leakage is a prominent issue as singular people have multiple MRI scans in the dataset. As the data was already supplied preprocessed by the teaching team there was no way to identify which brain belonged to which person and avoid the data leakage. The test set was used to find the accuracy after each epoch. Doing this in no way interferes with the training process or any hyperparameters of the model.

Example Brains:
Note, matplotlib automatically adds contrast colour when displaying image as the image dimention is only 1
![SampleBrains](../plots/sample_brains.png)

### Perceiver Modules
The perceiver consists of moultiple modules that can be effectively modeled with object oriented programming. The components of the perciever include cross-attention and multi head latent transformers (self attention). A generic attention block was made with the same layers as mentioned in the paper. Nomalisation, attention, linear layers and GELU. Skip connections were added between attention blocks as mentioned in the paper. Cross attention implements the attention layer with a single head between the latent and the image as specified in the paper. The latent transformer implements 8 latent heads with 4 total attention layers for self attention. The cross attention and Latent transformer together make one perceiver block in the model of which 3 were used.  

### Image Encoding
Due to the nature of the perciever it is permuation invarient and cannot directly draw on spatial structures as well as other models. To mitigate this the paper reccomended two options, fourier positional encodings or a learned array of embeddings. This project opts to use the learned embeddings. The embeddings are appended onto the colour dimention of the data.

### Classification
The classification layer reduces the 32x32x64 latent tensor to a 32x1 tensor for the binary output expected by BCELoss. It does this by passing through a fully connected layer followed by a mean on dimention 0 before another fully connected layer. The outputs are then passed through sigmoid before being returned. 

## Results
After 40 epochs the model achieved an accruacy of 57% on the testing set whcih is slightly better than random guessing. I attemoted many different hyperparameters to attempt to get the accuracy up. Doing more than 40 epochs made the model show signs of overfitting as the accuracy would go down and the loss would plateau. Learning rates greater than 0.0005 were shown to cause overshoot and an inability to converge. The model was shown to converge slowly over time but accuracy...

![Loss Plot for 40 Epochs](../plots/loss_plot_60_32x64.png) ![Loss Plot for 60 Epochs](../plots/loss_plot_60_32x64.png)

I suspect this is in due part to the small size of the dataset, the training set had slightly less than 10000 images which is considered quite small when training transformer models whcih require very large amounts of data, this is coulda also explain the overfitting after 40 epochs. Another reason could be the use of learned positional encodings over fourier encodings which might have made it difficult for the model to converge effectively. The performance is likely also in due part to the 32x128 latent array which I used. I had to use this size due to memory contraints on both my laptop and the p100 gpus on rangpur which I primarlity used to train the models with. Transformers are large complex models that benefit from scale which I was not able to provide. In the future I might try to incorporate data augmentation techniques to atrificially increase the size of the training set for potentially improve the model. 

## References
<a id="1">[1]</a>
Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., & Carreira, J. (2021, July). Perceiver: General perception with iterative attention. In International conference on machine learning (pp. 4651-4664). PMLR.
</br>
<a id="2">[2]</a>
Curt Tigges. (Aug 20, 2022). "The Annotated Perceiver." Medium. [Link](https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb)
