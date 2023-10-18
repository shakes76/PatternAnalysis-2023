---- documentation ---- 

# General Code Clean up 


# README OUTLINE 


Title 
Contents 
Introduction 
This project provides an implementation of the Improved UNET model for image segmentation to identify skin lesions. The dataset used for the model is from the ISIC 2018 Challenge. The Improved UNET architecture is from the paper by Isensee et al. 
Dataset
The ISIC 2018 Dataset was released as a resource for an automated image analysis tool development competition run by the International Skin Imaging Collaboration. The goal of the competition was to promote automated melanoma detection. 
The dataset consists of 2594 RGB images of skin lesions with corresponding ground truth masks that identify the skin lesion. The validation dataset cosnsits of …. Images. Finally, 1000 images are provided for testing and final results.
The dataset is available here.
An example input image and its mask are shown


Algorithm 
The UNET is a convolutional neural network (CNN) architecture used for biomedical image segmentation. The network forms a U-shaped structure when visualised, hence its name. It consists of a downsampling path on the left side where data is contracted to reduce spatial dimensions while encoding abstract essential features. This is done traditionally through stacks of convolutional and max pooling layers. The bottom of the “U” represents the bottleneck of the architecture where the network learns the most abstract features of the input. Finally, the right path forms the expansive path. This expansive path decodes data using transpose convolutional layers to recombine abstract representations with shallower features to retain spatial information for accurate segmentation. 
The UNET is also symmetrical between its downsampling and upsampling paths; each level (vertical depth) in the encoding path has a corresponding level in the decoding path, connected by skip connections to preserve information.
The Improved UNET builds upon a standard UNET by using complex aggregation blocks in its downsampling aggregation pathway, having additional layers and employing skip connections across different levels of the network. 
Model architecture 
Model photo -> with boxes around it 
The image above represents the Improved UNET architecture as depicted in the reference paper. It's important to note that the architecture utilizes 3x3x3 convolutions due to its 3D input nature. However, since the ISIC dataset comprises 2D images, all convolutions in our scenario will be 2D, albeit with a 3x3 filter. In the following sections, an in-depth examination of the architecture is provided, alongside references to the respective modules in the code.
Downsampling path:
•	Overview: The downsampling path inputs the image into several convolutional blocks and context modules. Each downward step doubles the number of channels while halving the spatial dimensions of the image. This structure allows the network to learn and encode more complex features at different levels of the abstraction. 
•	Components:
•	3 x 3 convolution: This performs a 2D convolution with a kernel size of 3x3 with stride one to output the same shape as as the input. This is connected to an instance normalisation layer and a leaky ReLU activation. This is implemented as ‘StandardConv’ in modules.py
•	3x3 convolution with stride 2: Performs a 3x3 convolution with a stride of 2. This reduces spatial volume as we go deeper into the network. 
•	Context module: In each context module block, there are two convolutional layers with a dropout layer in between to avoid overfitting. 
Upsampling Path
•	Overview: 
The right side of the U-Net architecture receives the left side's output, upscales it, and then concatenates it with earlier layers' outputs (skip connections). These concatenated outputs are further processed through localization modules, repeating this process up through the network, gradually increasing the spatial dimensions while decreasing the feature channels. At the first level, a 3x3 convolution with 32 filters is used instead of a localization module. These skip connections help in recovering spatial information lost during downsampling, which is crucial for precise segmentation in the upsampling part of the network.
Components:
•	Upsampling Module: the upsampling module upsamples input via a convolutional transpose 2d, then goes through a vconvolution again. This doubles the input shape.
•	Localisation Module: contains two convolutional blocks 

Segmentation Components:
In the architecture, the Segmentation Level area processes the output from the localization module in the third and second levels through a segmentation layer. The third level's segmentation output is upscaled to match the second level's shape, followed by an element-wise sum, acting as a short skip connection. This process is repeated with the first level's 3x3 convolution output, involving upscaling and element-wise summing. The final short skip connection is passed through a sigmoid function for a final output, as it deals with a single class, producing a 2D segmented output of dimensions 256x256x1.
Segmentation Layer: Executes a 2D convolution with a 1x1 kernel size and zero padding, ensuring the input and output shapes remain identical, followed by instance normalization and a leaky ReLU activation (negative slope of 0.01).
Upscale: This layer doubles the dimensions of the input for the output.
Sigmoid: Conducts a 2D convolution with a 1x1 kernel size, retaining the input and output shapes, and then applies a sigmoid activation function for the final output.


Training
•	Dice Loss 
•	Parameters and their implementation 
Hyperparams
Training results
•	With dice loss etc.

Model Evaluation

How To Use

Reproduction 
Dependencies 
Dataset Link
References 

