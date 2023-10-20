
# OASISVQ: Enhancing Brain Image Generation with VQVAE

OASISVQ is a project dedicated to advancing brain image generation using the Vector Quantized Variational Autoencoder (VQVAE) deep learning model. This README provides an overview of the project and its goals.

# Ploblem Overview
Brain image generation is a crucial task in neuroimaging research, aiding in the study of brain structure, function, and various neurological conditions. OASISVQ aims to utilize the capabilities of VQVAE, a powerful variant of Variational Autoencoders, to enhance the generation of brain images from the OASIS Brain dataset.

# Objectives
- Implement a VQVAE model for brain image generation.
- Achieve a "reasonably clear image" with a Structured
- Similarity (SSIM) score exceeding 0.6.


# Dataset
Data were provided by OASIS in [this resouce](https://www.oasis-brains.org/#data) [6] [7].  The OASIS Brain dataset is used as the foundation for training and evaluating the VQVAE model. This dataset encompasses structural and functional MRI scans, covering a wide range of subjects, including healthy individuals and those with neurological conditions. OASIS 1 and 2 were both used in this problem [6] [7].

---

# Model Architecture

##### VQ-VAE Definition
A VAE is a type of generative model that falls under the umbrella of autoencoders. It consists of an encoder, a decoder, and a latent space where the data is represented in a compressed form.[5]  VQ-VAE is an extension of the basic VAE architecture, and it incorporates vector quantization to improve the quality of generated samples. In a VQ-VAE, the encoder produces discrete latent variables, and a separate codebook (dictionary) is used to quantize these variables

##### Overview
 The VQ-VAE model archictecture consists of three parts, the encoder, the Vector Quantization layer, and the decoder. 
 
<p align="center">
	<img src="Images/vqvae2.png" width=800>
	<p>
    <em>Left: The Encoder, Middle: the Vector Quantization layer, Right: the Decoder</em>
		



</p>

######Encoding layer

Summarised from [this paper](https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a), the encoder layer takes in an image noted as x with the following parameters: 
- n: batch size
- h: image height
- w: image width
- c: number of channels in the input image

######Vector Quantization layer

The VQ layer operates in six key steps, illustrated in Figure 2. Initially, a Reshape combines dimensions into vectors. Distances are then calculated between these vectors and those in the embedding dictionary, yielding a matrix. The Argmin step finds the index of the closest dictionary vector for each input vector. Subsequently, the closest dictionary vectors replace the original vectors. Reversing the Reshape operation, vectors are restored to the shape (n, h, w, d). Since backpropagation can't flow through Argmin, gradients are approximated by copying them from the quantized vector (z_q) back to the original vector (z_e). Despite not directly minimizing the loss function, this process transmits information for training.

######Decoding layer
 The decoder in a VQ-VAE is typically a neural network responsible for generating data samples from the latent representations produced by the encoder. In the context of a VQ-VAE, the decoder is responsible for reconstructing the data, such as images or audio, from the discrete latent variables learned during the encoding process.


---

# Models and HyperParameters 

###### Generated Images 

#### Pixel-Cnn
PixelConvLayer:

The PixelConvLayer is a custom layer used in your PixelCNN model. It is designed to implement pixel-wise convolutions with specific mask types, which are commonly used in autoregressive models like PixelCNN for image generation.

| Attributes | Information |
|------------|-------|
mask_type: |This attribute specifies the type of mask applied to the layer. In autoregressive models, different masks are used for different layers to ensure that each pixel in the output depends only on previously generated pixels. There are two common mask types: "A" and "B."
conv: | This attribute is an instance of the Keras Conv2D layer, which is used to perform convolution operations.
build(input_shape): | This method is responsible for building the layer by initializing the kernel variables. It creates a mask based on the specified mask_type, which restricts the convolution's receptive field.
call(inputs): | This method performs the forward pass of the layer. It applies the convolution operation with the configured mask to the input tensor. The mask ensures that each output pixel depends only on previously generated pixels according to the autoregressive constraints.


The ResidualBlock is a building block used in your model. It's a common architectural component in many deep learning models, and in this context, it's used within the PixelCNN model.

| Attributes | Information |
|------------|-------|
filters: | This attribute specifies the number of filters (output channels) used in the convolutional layers within the block.
Methods:
call(inputs): | This method performs the forward pass of the residual block. It consists of several layers:
conv1: A 1x1 convolution layer with filters filters and ReLU activation.
pixel_conv: |An instance of the PixelConvLayer, which performs convolution with a specific mask type.
conv2: | Another 1x1 convolution layer with filters filters and ReLU activation.
The output of this block is obtained by adding the input inputs to the result of the final convolution. This is a common practice in residual networks to help with gradient flow and mitigate vanishing gradient problems.
In summary, the PixelConvLayer implements pixel-wise convolutions with specific masks for autoregressive image generation, and the ResidualBlock is a building block with convolutional layers and residual connections used to learn features in the PixelCNN model. These components contribute to the model's ability to generate images with autoregressive constraints.

---

###### Generated Images 

<p align="center">
	<img src="Images/Encoded.png" width=800>
</p>


<p align="center">
	<img src="Images/Reconstructed.png" width=800>
</p>






# Loss Functions and Structual Similarity 
Reconstruction Loss (reconstruction_loss): This loss measures the dissimilarity between the original input data and the data reconstructed by the VQ-VAE model. It is computed as the mean squared difference between the input and the reconstructed output. The goal is to minimize this loss, ensuring that the VQ-VAE can accurately reconstruct the input data.

<p align="center">
	<img src="Images/ VQVAE.png" width=800>
</p>

Sparse Categorical Crossentropy Loss (keras.losses.SparseCategoricalCrossentropy): This loss function is commonly used for models that generate discrete outputs, such as image generation with PixelCNN. It measures the dissimilarity between the predicted probability distribution (logits) and the ground truth discrete values (in this case, the codebook indices). The goal is to minimize this loss, encouraging the model to generate pixel values that match the target distribution.

<p align="center">
	<img src="Images/PixelCNN.png" width=800>
</p>

The Structural Similarity Index (SSIM) is a metric used to measure the similarity between two images. It was designed to assess the perceived quality of images by considering both structural information and luminance. SSIM is widely used in image processing and computer vision tasks, including image compression, denoising, and quality assessment. 

<p align="center">
	<img src="Images/Ssim.png" width=800>
</p>

| SSIM Range | Interpretation |
|------------|----------------|
| 1.0        | Perfect Match   |
| 0.7 - 0.99 | Good            |
| 0.5 - 0.69 | Fair            |
| < 0.5      | Poor            |

An SSIM index of 0.70385885 is considered good. It suggests that the two images being compared have a strong structural similarity, and the differences between them are relatively small and likely not easily discernible to the human eye. This level of similarity is generally desirable in various image processing applications, such as image compression, denoising, or image quality assessment.

---

# Set Up and Example Usage 
## Requirements and Dependency
matplotlib >= 3.5.2
numpy >= 1.21.5
requests >= 2.28.1
tensorflow >= 2.10.1
tensorflow-probability >= 0.14.0
python >= 3.7.13

### Installments
Installing conda is reccomended for the depedencies, especially PyTorch. Please follow the custom environment set up for project requirements.


If you prefer do it manually, use this example:
1. Create a conda environment and name it eg. my-torch 


```
conda create -n my-torch python=3.7 -y
```
2. Activate the new environment
``` 
conda activate my-torch
```
3. Inside environemnt my-torch, install PyTorch:

```
conda install python=3.6 pytorch torchvision matplotlib pandas -c pytorch
```
### Package Overview
__predict.py__
Predict shows example ussage of the trained model. 
Requires a trained model saved to files from running train.py
```
python predict.py

```
__dataset.py__
Dataset holds the class for the OASIS dataset attributes, downloads the data to your device and preprocesses the data. Dataset is called from train.py and does not need to run on its own.

__modules.py__
Models.py contains the VQ-VAE model [3] and the Pixel CNN [4]. Modules.py should be called from predict.py or train.py. 

__train.py__
Train.py loads the dataset onto the compuer, and trains the VQ-VAE in 30 epochs and saves the history of the model for loss graphing. Next, the weights of the VQ-VAE are saved. These weights are implemented into a temporary VQ-VAE model to train the PixelCNN. The PixelCNN's weights are saved into the package folder. Training loss and plots are graphed and saved. 

To load the weights of the VQ-VAE into another program or software, simply follow these steps. 
1. Run the train.py module
```
python train.py

```
2. In your python file, run the following commands
```
from dataset import *
from modules import *

models_directory = <directory_of_saved_model>
vqvae_weights_filename = <filename_of_vqvae_weights>

vqvae_trainer.load_weights(models_directory + vqvae_weights_filename)

```
[2]
__similarity.py__
Similarity.py runs the structural similairty score. This file can be run on its self and plots the structural similarity. [3]
```
python similarity.py
```



### gitignore 
__hyperparemeters.py__
The hyper-parameters for this model are specified below and incorporated throughout the problem. For improvement it may be better to save the hyperparameters as a seperate python file 

---
## References
[1] https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a
[2] https://keras.io/examples/generative/vq_vae/
[3] https://en.wikipedia.org/wiki/Structural_similarity
[4] https://keras.io/examples/generative/pixelcnn/
[5] https://arxiv.org/abs/2101.08052
[6] OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382. https://www.oasis-brains.org/#data
[7] OASIS-2: Longitudinal: Principal Investigators: D. Marcus, R, Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.https://www.oasis-brains.org/#data



