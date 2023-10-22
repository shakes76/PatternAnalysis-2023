# COMP3710 StyleGAN2 Brain Image Generation
An implementation of StyleGAN2 for generating images of the human brain based on the [OASIS brains](https://www.oasis-brains.org/) dataset.

## Table of Contents
- [COMP3710 StyleGAN2 Brain Image Generation](#comp3710-stylegan2-brain-image-generation)
  - [The Problem - *Reasonably Clear Image Generation*](#the-problem---reasonably-clear-image-generation)
  - [Why a GAN?](#why-a-gan)
  - [StyleGAN & StyleGAN2](#stylegan--stylegan2)
      - [*Figure 1 - Traditional GAN vs StyleGAN Generation Architecture*](#figure-1---traditional-gan-vs-stylegan-generation-architecture)
      - [*Figure 2 - Revised StyleGAN2 Architecture*](#figure-2---revised-stylegan2-architecture)
  - [Dataset](#dataset)
  - [Implementation](#implementation)
    - [Dataloader Implementation](#dataloader-implementation)
      - [Dataloader Pre-processing](#dataloader-pre-processing)
    - [Predict/Image Generation Implementation](#predictimage-generation-implementation)
  - [Training & Results](#training--results)
    - [Figure 3 - Training Results (w_dim = 512) and (training epochs = 300)](#figure-3---training-results-w_dim--512-and-training-epochs--300)
  - [Code Structure](#code-structure)
  - [Usage/Try it yourself](#usagetry-it-yourself)
    - [Requirements](#requirements)
    - [Getting Started](#getting-started)
    - [Training a Model](#training-a-model)
    - [Generating New Images](#generating-new-images)
  - [Where to Next?/Discussion](#where-to-nextdiscussion)
  - [References & Acknowledgements](#references--acknowledgements)

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
revised StyleGan2 architecture can be viewed in Figure 2. The revised architecture 
removed the mean normalisation step in the *AdaIN* layer in order to prevent the 
blob-like artifacts from appearing in generated images. Additionally, in order to 
produce more predictable results, the addition of the bias and the gaussian noise was 
moved so it is added after the data is normalised. The StyleGAN2 also introduces a new 
*Weight Modulation* step, which combines the modulation and the convolution 
operations so the convolutional kernel weights are scaled with the style vector. This 
allows for full controllability whilst still removing the blob-like artifacts. 

#### *Figure 2 - Revised StyleGAN2 Architecture*
![Revised StyleGAN2 Architecture](./assets/styleGAN2%20architecture.PNG)

## Dataset
The dataset utilised for training the model is the [OASIS brains](https://www.oasis-brains.org/). You can find a download link for the data [here](https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA). After downloading the data you will notice that it has the following directory structure -

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
    |   |   |── data
    │   ├── keras_png_slices_seg_train  
    |   |   |── data
    │   ├── keras_png_slices_seg_validate    
    |   |   |── data
    │   ├── keras_png_slices_test  
    |   |   |── data
    │   ├── keras_png_slices_train  
    |   |   |── data
    │   ├── keras_png_slices_validate  
    |   |   |── data

The data is now ready for use with the dataloader within this project.

## Implementation
As mentioned, the StyleGAN2 model was implemented with the objective to generate reasonable clear 
images of the brain. The models utilised within this project (found in the `modules.py` directory) 
were based on the original [StyleGAN](https://arxiv.org/pdf/1912.04958.pdf?ref=blog.paperspace.com) & [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf) papers as well as this [light weight StlyeGAN2 implementation](https://blog.paperspace.com/implementation-stylegan2-from-scratch/) found online. This StyleGAN2 implementation was then modified and re-tuned 
to run on the OASIS brains dataset. One critical parameter which had to be re-tuned was the latent 
space dimensions. The original paper utilised a style vector ($w$) dimension of 512. As such the 
models were modified to accommodate this new latent dimension. 

To implement the algorithm I started by first writing the dataloader for the OASIS brains dataset, as 
described [below](#dataloader-implementation). I tested the dataloader was running correctly on both 
my local machine and on the University Rangpur High Performance Cluster (HPC). After verifying that 
the dataloader was working, I modified the models to work with the new dataloader. 

The next phase of implementing the solution was to develop the training loop. To train the GAN, a 
batch of *real* images is first fetched from the data loader. Then random style vectors ($w$) and gaussian 
noise vectors were generated, and passed into the Generator model to obtain a batch of fake generated 
images. Using the Discriminator model each of these fakes would then be assessed to identify if they 
were classified as real or fake. After this the original batch of real images would also be passed 
through the discriminator and the loss gradients were then calculated. After computing the loss, back 
propagation occurs and the weights of the models are updated. Lastly, as per the StyleGAN2 paper it 
is recommended that path length regularisation is implemented, the `path length penalty` metric is 
added to the loss every 16 iterations in order to ensure the *"smoothness"* of the latent space 
manifold being learnt. This process is then repeated for 300 epochs of training. Lastly, training 
loop utilised `tqdm` to log the progress of the training.

The last component of the implementation was to create a image generation script so visual inference 
can be done on the latent space. The implementation details regarding this model can be found [here](#predictimage-generation-implementation).

### Dataloader Implementation
The implementation of the dataloader for the OASIS dataset can be found in the `dataset.py` module. 
To load and prepare the OASIS dataset for training, the `generateDataLoader` function is utilsed. 
This data loader will load in the brain slices `train`, `test` & `validate` sets from the dataset. 
For the purposes of this StyleGAN2 implementation, only the `trainset` will be utilised since there 
is no technique to utilise the `validate` and `test` data sets within the StyleGAN2 architecture. 

#### Dataloader Pre-processing
The OASIS brains dataset provided, is already pre-processed, as such no additional pre-processing was 
*strictly* required, however some basic pre-processing was still included in the event that a different version of the dataset was being utilised.

Since the size of the dataset is fairly small (9k images), to the reduce the risk of over-fitting 
dataset augmentation was implemented. Upon import a random horizontal flip transform was applied to 
the images to account for this. Additionally, upon import the images were all normalised.

### Predict/Image Generation Implementation
The `predict.py` functionality was utilised to generate a grid of images from the trained latent space of the Generator model. Every time this script is executed, a new subset of randomly generated images will be displayed in the grid since the new randomised style vectors ($w$) are being used to generate the outputs.

## Training & Results
As mentioned above, this StyleGAN2 implementation has been trained on the OASIS brains dataset. A snapshot of some example input training images from this dataset can be seen in [Figure 3](#figure-3---training-input-sample-images-from-the-oasis-dataset). 

#### *Figure 3 - Training Input Sample Images (From the OASIS Dataset)*
![Training Input Sample Images](./assets/trainingImageSample.png)

The final results after training for 300 epochs, with a style vector dimension of 512 can be seen in Figures [4](#figure-4---training-results-w_dim--512-and-training-epochs--300) & [5](#figure-5---training-results-alternative-w_dim--512-and-training-epochs--300). To generate these images, random gaussian noise vectors were created and passed through the mapping network to obtain unique style vectors $w$. These randomised vectors were then used to create the batch of 64 images shown in the grid. As such each of the brain images within the grid is generated from a different style vector.

#### *Figure 4 - Training Results (w_dim = 512) and (training epochs = 300)*
![Training Results](./assets/generatedImages2.png)

#### *Figure 5 - Training Results Alternative (w_dim = 512) and (training epochs = 300)*
![Training Results](./assets/generatedImages.png)

Overall, the above results depict *reasonably clear* images of the brain and as such, the objective of the task has been achieved. The results have appeared fairly consistent with their not being bias toward a particular feature/style. I was able to consistently reproduce similar results each time I trained the StyleGAN2 model.

The training progression of the StyleGAN2 can be visualised by viewing the sample images generated at different epochs of training time. Below [Figure 6](#figure-6---training-progression-of-the-stylegan2-w_dim--512-and-training-epochs--300) shows that there is an improvement in clarity as the number of training epochs increases.

#### *Figure 6 - Training Progression of the StyleGAN2 (w_dim = 512) and (training epochs = 300)*
![Training Progression 50-250 epochs.](./assets/trainingProgression.PNG)

Lastly, to ensure that the model is training correctly, the loss plots were analysed. As demonstrated by [Figure 7](#figure-7---training-loss-plot-w_dim--512-and-training-epochs--300), the loss of the generator and the discriminator models converge at approximately 0. This indicates that the GAN is being trained correctly and that there is not a bias toward one of the models in particular. 

#### *Figure 7 - Training Loss Plot (w_dim = 512) and (training epochs = 300)*
![Training loss plot after 150 epochs](./assets/trainingLossPlotAvgPerEpoch_150epochs.png)
*Note: that there was no additional variation after epoch 150 so the domain of the graph is [0:150]*

## Code Structure
Here is a summary of how the code base is structured.
- `assets/` -> a directory containing all of the assets for the README.md file.
- `dataset.py` -> a data loader module for the OASIS brains dataset
- `train.py` -> a training script for the StyleGAN2 model
- `predict.py` -> an inference script which can be used to generate new images from the trained generator
- `config.py` -> a hyper-parameter tuning configuration file to control the outputs of training and prediction
- `environment.yml` -> a YAML file which details the contents of the conda environment
- `README.md` -> (the file you are currently reading) the documentation surrounding the project

## Usage/Try it yourself
### Requirements
The main dependencies required for this project are as follows:

    - python 3.11.5
    - pytorch 2.1.0
    - torchvision 0.16.0
    - matplotlib 3.7.2
    - tqdm 4.66.1

A full list of the dependencies I have utilised can be seen in my conda environment `.yml` file [available here](environment.yml).

### Getting Started
To get started and try the code on your own machine, start by going through the following steps:
1. Clone this repository to your local machine. 
2. Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) if you do not already have it installed on your machine.
3. Change to the `PatternAnalysis-2023/recognition/46990480_StyleGan2` directory & run the following commands to create the required conda environment from the provided `.yml` file.
    ```
    conda env create --name {ENV_NAME} --file=environment.yml
    ```
    *NOTE: change the `ENV_NAME` to any environment name you wish.*
4. Activate your newly python environment by running -
    ```
    conda activate ENV_NAME
    ```
5. Download the dataset and configure the folder structure as detailed above in the [Dataset](#dataset) section.
6. Create a new folder to store the outputs of the models `./PatternAnalysis-2023/recognition/46990480_StyleGan2/Models`
6. Your workspace is now configured to run the provided code.

### Training a Model
1. Configure the `config.py` script with your desired dataset & save paths. Here you can also tweak any of the other training hyper-parameters, if you so wish. I suggest setting the `modelName` to something unique if you are running the script repeatedly, this will prevent the outputs from `train.py` & `predict.py` from overwriting the previous results.
2. Run the training script with the command -
    ```
    python train.py -run="TRUE"
    ``` 
    This will create a folder `./saved_examples_{MODEL_NAME}/` where sample images after every 50 epochs of training will be saved.
3. After training is complete, a training loss for the discriminator and generator networks will be saved here for inference -
    ```
    {output_path}saved_examples_{modelName}/trainingLossPlotAvgPerEpoch.png
    ```
    Where `output_path` and `modelName` are defined in `config.py`. Additionally,
    a new folder called `./Models` will also be created where you can find the exported Generator, Discriminator & Mapping Network trained models.

### Generating New Images
The `predict.py` script allows you to generate a grid of new images based on the trained Generator & Mapping network models exported by `train.py`. After training your models run the command `python predict.py` to generate batch of sample images. Sample images are generated by creating a random noise vector and passing it through the Mapping Network model to obtain a randomised style vector $w$. This style vector is then used to create the image.

Use the following command line arguments to customise the output of the `predict.py` plots -
- `-load_path_mapping` -> Specify a specific load path for the Mapping Network model. Default = `{save_path}MAPPING_NETWORK_{modelName}.pth`
- `-load_path_generator` -> Specify a specific load path for the Generator model. Default = `{save_path}GENERATOR_{modelName}.pth`
- `-num_output` -> Specify the number of output images to show within the grid. Default = `64`
- `-plt_title` -> Specify the title for the generated images output grid. Default = `"Generated Images"`
- `-train_models` -> Train the model before generating the plots. Note: this will override any provided `-load_path_mapping` and `-load_path_generator` arguments and use the parameters outlined in the `config.py` file.

### All in One (Training, Inference & Image Generation)
The `predict.py` script can be used as a driver for the training and image generation stages described above. In order to train a model, generate the loss plot & generate a batch of images in the same operation, use the following command:
```
python predict.py -train_models=TRUE
```

## Discussion
To further improve the results of the output, I would like to compute additional metrics to gain a better understanding of how the models are training. Specifically, I would like to try implementing a [UMAP](https://umap-learn.readthedocs.io/en/latest/) to better visualise the latent space within the trained generator model.

## References & Acknowledgements
- https://www.oasis-brains.org/
- https://arxiv.org/abs/1812.04948
- https://arxiv.org/abs/1912.04958
- https://blog.paperspace.com/implementation-stylegan2-from-scratch/#models-implementation
