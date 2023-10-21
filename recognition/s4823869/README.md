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