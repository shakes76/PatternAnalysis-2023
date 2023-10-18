# COMP3710 StyleGAN2 Brain Image Generation
An implementation of StyleGAN2 for generating images of the human brain based on the [OASIS brains](https://www.oasis-brains.org/) dataset. 

## Contents

## The Problem - *Reasonably Clear Image Generation*
The sophistication and complexity of the human brain has fascinated scientists for hundreds
of years. With the recent momentum surrounding generative AI, it is time to harness this
technology and produce images of brains. The objective of this project was to design a
StyleGAN2 implementation with the purpose of generating *reasonably clear* 256x256px images 
of the human brain. 

## Why a GAN?
A Generative Adversarial Network (or GAN) is an deep learning model comprised of two 
convolutional networks, a Generator network and a Discriminator network. The Generator 
network attempts to generate 'fake' images from random noise input in order to trick the 
discriminator network into thinking that the produced image is 'real'. Mathematically 
speaking, the two networks are playing a min-max game whereby we are trying to maximise the 
loss of the discriminator and minimise the loss of the generator.

## StyleGAN & StyleGAN2
The style-based GAN (StyleGAN) architecture, developed by various NVIDIA researchers 
in 2020, is a GAN architecture capable of generating high fidelity realistic images of 
from it's latent space. The key difference between the StyleGAN and a conventional GAN 
architecture is that the StyleGAN prevents feature entanglement by mapping the $z$ 
latent vector into a style vector latent space $w$ as highlighted in [Figure 1](#figure-1---traditional-gan-vs-stylegan-generation-architecture). This is achieved by 
passing the latent vector $z$ through a fully connected Mapping Network to produce the 
style latent vector $w$. This intermediate style vector latent space $w$ results in 
the an untangled feature distribution in the latent space, resulting in easier 
training of the Generator network. As depicted in Figure 1, the latent vector $w$ is 
then passed into the Generator ($g$) in the Adaptive Instance Normalization (*AdaIN*) 
layer. Additional gaussian is also added after each of the convolution operation, this 
stochastic variation improves the finer details within an image making them look even 
more realistic. The two main issues with the StyleGAN (and the subsequent motivation 
for the StyleGAN2 architecture), were that blob-like artifacts were appearing in the 
images and detailed features had a strong location preference due to the progressive 
growing nature used within the generator network.

#### *Figure 1 - Traditional GAN vs StyleGAN Generation Architecture*
![Traditional GAN vs StyleGAN Generation Architecture](./assets/StyleGAN%20general%20architecture.PNG)

To resolve these issues, the StyleGAN2 architecture was developed. The details of the 
revised StyleGan2 architecture can be viewed in Figure x. The revised architecture 
removed the mean normalisation step in the *AdaIN* layer in order to prevent the 
blob-like artifacts from appearing in generated images. Additionally, in order to 
produce more predictable results, the addition of the bias and the gaussian noise was 
moved so it is added after the data is normalised. The StyleGAN2 also introduces a new 
*Weight Modulation* step, which combines the modulation and the convolutions 
operations so the convolutional kernel weights are scaled with the style vector. This 
allows for full controllability whilst still removing the blob-like artifacts. 

TODO: Add progressive growing
To resolve the strong location preference occurring within the generated images, 

StyleGAN2 replaced the progressive growing 

#### *Figure 2 - Revised StyleGAN2 Architecture*
![Revised StyleGAN2 Architecture](./assets/styleGAN2%20architecture.PNG)

## Dataset
The dataset utilised for training the model is the [OASIS brains](https://www.oasis-brains.org/). You can find a download link for the data [here](). After downloading the data you will notice that it has the following directory structure -

├── keras_png_slices_data  
│   ├── keras_png_slices_seg_test  
│   ├── keras_png_slices_seg_train    
│   ├── keras_png_slices_seg_validate    
│   ├── keras_png_slices_test  
│   ├── keras_png_slices_train  
│   ├── keras_png_slices_validate  

Where the images are located within each of the sub-folders respectively. 

In order to  utilise this dataset with this project, we must slightly modify the structure of the data so it is compatible with the [`ImageFolder` api](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).

To do this move all of the images within each directory into an additional sub-folder called data. After you do this, your dataset file hierachy should look like this -

├── keras_png_slices_data  
│   ├── keras_png_slices_seg_test    
|   |   ├── data  
│   ├── keras_png_slices_seg_train    
│   ├── keras_png_slices_seg_validate    
│   ├── keras_png_slices_test  
│   ├── keras_png_slices_train  
│   ├── keras_png_slices_validate 

The data is now ready for use with the dataloader within this project.

## Requirements

The main dependencies required for this project are:

    - python 3.11.5
    - pytorch 2.1.0
    - matplotlib 3.7.2
    - tqdm 4.66.1

A full list of the dependencies I have utilised can be seen in my conda environment .yml file [available here]().
### Code Structure

## Model Implementation
## Try yourself
To get started and try the code on your own machine use the following steps:
       
1. Download the dataset and configure the folder structure as detailed above in the [Dataset](#dataset) section.
2. Configure the config.py script with your dataset & save paths. Here you can also tweak any of the other parameters associated with the model.
3. Run the training script with the command `python train.py`. This will create a folder `./saved_examples_{MODEL_NAME}/` where sample images after every 50 epochs of training will be saved. A copy of your training loss plot will also be saved here to this folder. A few folder called `./Models` will also be created where you can find your exported trained Generator, Discriminator & Mapping Network models.
4. Run inference on your trained models and generate some images from your trained generator and mapping networks by running the command `python predict.py`. By default this will create a plot with 64 images generated by the Generator model.

### predict.py
The `predict.py` script also serves as a command line tool to run inference on your model & train it at the same time.

## Training & Results

## Where to Next?

## References & Acknowledgements
- https://www.oasis-brains.org/