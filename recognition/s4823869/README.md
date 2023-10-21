# **StyleGAN for OASIS Brain Dataset**

**Name - Yash Mittal**

**UQ ID – 48238690**
## <a name="_overview"></a>**Introduction**

Welcome to the StyleGAN project designed to generate "reasonably clear images" from the OASIS brain dataset. This model is a powerful implementation of the cutting-edge Generative Adversarial Network (GAN) architecture. This project goes beyond the basics of GANs and includes several innovative features to enhance the image generation process.

## <a name="_key_model_features"></a>**Key Model Features**

The StyleGAN for the OASIS brain dataset boasts a range of advanced features:

*1. Progressive Training*

Progressive training is a fundamental aspect of my model. It starts with low-resolution images (4x4) and gradually increases the resolution by adding new layers. This approach not only accelerates the training process but also ensures stability. The progressive training technique is supported by both the generator and discriminator and implemented in the train.py file.

*2. Adaptive Instance Normalization (AdaptiveInstanceNorm)*

I've implemented AdaptiveInstanceNorm to normalise the style feature of the generator. This feature allows us to effectively modulate and manipulate generated images based on the style factor 'w.' The AdaptiveInstanceNorm class in the modules.py file is responsible for this technique.

*3. Stochastic Variation with Noise Inputs*

I've introduced stochastic variation through scale/resolution-specific Gaussian noises to add depth and fine-grained details to the generated images. The NoiseInjection class in the modules.py file handles this feature. The learnable scaling factor ensures that specific details, such as hair and freckles, are faithfully represented.

*4. Weight-scaled Convolution*

My model employs weight-scaled convolution to ensure the stability of the training process. This technique involves normalising the weights of convolutional layers based on input channels and kernel size. The WeightScaledConv2d class in the modules.py file is responsible for this weight scaling. It's important to note that this feature is applied to both the generator and discriminator, ensuring consistent results.

*5. Spectral Normalization*

In addition to weight scaling, my model incorporates spectral normalisation to enhance training stability further. Spectral normalisation is applied to the discriminator's layers to help maintain equilibrium during the adversarial training process.

*6. Self-Attention Mechanism*

I've introduced a self-attention mechanism within the model to capture long-range dependencies in images. This component allows the generator to focus on essential image regions, resulting in higher-quality outputs.

*7. Conditional Generation*

The StyleGAN model supports conditional image generation. By providing additional input, you can control the characteristics of the generated images, allowing for greater flexibility and customisation.

![RelationalDatabaseVsNoSQLDatabase.png](RelationalDatabaseVsNoSQLDatabase.png)

*Fig 1: Shows the original demonstration of the Style-Based generator* **[1]**
#
# <a name="_code_structure"></a>**Code Structure**

1. *dataset.py:*

*Purpose:* This script prepares a dataset from the OASIS dataset for training and testing, including data transformation and providing data loaders. It also displays sample images from the dataset.

*Key Components:*

- CustomImageDataset class: Loads and organises image data for training and testing.
- Loader functions to load data.


2. *modules.py:*

*Purpose:* This module defines neural network modules for a Generative Adversarial Network (GAN), including the Generator and Discriminator networks.

*Key Components:*

- StyleMapping class: Maps the latent noise vector to the intermediate style vector used by the Generator.
- PixelwiseNormalization class: Performs pixel-wise normalisation.
- WeightScaledLinear class: Linear layer with weights scaled for training stability.
- WeightScaledConv2d class: Convolutional layer with scaled weights.
- AdaptiveInstanceNorm class: Performs adaptive instance normalisation for style transfer.
- ConvolutionBlock class: Represents a block of convolutional layers in the generator.
- NoiseInjection class: Injects noise into the input tensor.
- GeneratorBlock class: Represents a block in the generator with adaptive instance normalisation and noise injection.
- Generator class: Defines the generator network for image generation.
- Discriminator class: Defines the discriminator network for image discrimination.


3. *train.py:*

*Purpose:* This script trains a Progressive Growing GAN (PGGAN) using the provided modules. It implements both the generator and discriminator training steps, including the calculation of the gradient penalty for the Wasserstein GAN with Gradient Penalty (WGAN-GP) loss.

*Key Components:*

- Training loop for both the discriminator and the generator.
- calculate\_gradient\_penalty function: Calculates the gradient penalty for enforcing the Lipschitz constraint.
- Main training loop for progressive training.


4. *predict.py:*

*Purpose:* This script generates images using a pre-trained GAN generator and displays them in a 3x3 grid. The generated images are saved in the 'output\_images' directory and displayed on the screen.

*Key Components:*

- Loading a pre-trained generator model.
- Generating a specified number of sample images.
- Preparing and visualising the generated images in a 3x3 grid.
- Saving the generated image grid to a file.


These files work together to prepare a dataset, define the GAN architecture, train the GAN, and generate images using the pre-trained generator. The training is performed progressively, starting from a lower resolution, and gradually increasing the image size. The generated images are saved and displayed for evaluation.
