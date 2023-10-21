**Introduction:**

In this report, a VQVAE will be developed in Pytorch to generate new images using the OASIS dataset (grayscale MRI images of the brain).
A VQVAE (Vector-Quantised Variational Autoencoder) is a neural network architecture, which learns discrete latent space representations of the input. It reconstructs the original input from these representations. Discretising the latent space gives VQVAEs many advantages over the conventional VAEs, making the representations more compact and interpretable. Using an autoregressive neural network to learn the space of discrete representations, new plausible outputs are constructed by sampling from this learned space.

**Hyperparameters:**

The hyperparameters were tuned throughout the report. The learning rate which yielded the best outcome for the VQVAE was found to be 5e-4, and the batch size, 128. Scheduling was considered however wasn’t necessary for the models.
The number of embeddings was chosen to be 256, and the initial embedding dimension as 32. The embedding dimension, however, was increased, as the low embedding dimension was found to be insufficient in capturing the fine details of the dataset, bottlenecking the model. This caused the loss plateauing at roughly 0.3, with the model not yet achieving the desired sharpness due to this limiting factor.

**Results:**

The results of the report are shown below, the model generates plausible images, albeit extremely blurry, with much room for improvement. After 60 epochs, the VQVAE was able to produce accurate, albeit blurry, reconstructions. A graph of the losses is shown below. The individual losses (codebook loss, commitment loss and reconstruction loss) were checked numerous times to ensure expected behaviour. 
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/fac1f5a9-42ce-4461-8418-52253bac6bf6)

This VQVAE produced the outputs shown below:
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/2821e6e7-ee87-4b0e-9ae9-955b864f8df3)

 
These results demonstrated that the model was not overfitting and should be trained for more epochs. The loss, however, plateaued at roughly 0.3, which was an indication that the model was not complex enough to capture the detail of the dataset. The Pixel CNN was then trained for 25 epochs, with the results as shown below (Top: generated indices, Left: bottom half of the brain generated, Right: entire image generated):
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/a4f2a8d4-430a-4f4e-bd5d-5ce9487003c1)
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/944c35c0-4cee-4d98-90fa-32a8bcfb5a4b)
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/37bd9bd0-6869-4af4-8982-6a1405dc295a)

   
These results demonstrated the model working as intended, even after just 60 epochs and 25 epochs respectively. The loss graph for the PixelCNN is shown below (Note the x-axis for the validation loss represents the total number of samples, not the epochs)
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/727e40c9-7079-41d8-8b65-11f1dac486ba)


Overfitting was also avoided here, as can be seen from the graph, however the PixelCNN was also limited due to the VQVAE (which at this point had only trained for 60 epochs).
This led to the number of hidden channels in the encoder and decoder being increased slightly, as well as the embedding dimensions (if necessary, the number of embeddings can also be increased).
The new model performed far better than the old, (however did not save correctly due to failure of the technology used), and was able to complete the task. The model provided is able generate far clearer and more accurate image after approximately 80 and 50 epochs (without overfitting).
If even more accurate results are required, the overall number of parameters (particularly embedding dimension, number of embeddings), can be increased further.

**How to Use:**

In order to run this file, the environment must have to following installed:
-	Pytorch
-	TQDM
-	Numpy
-	PIL
-	Matplotlib
Create a new conda environment with python 3.8 then run the command ‘pip install library’ for each of the libraries above in the command terminal to install any missing libraries. The OASIS dataset will also need to be downloaded and placed in a directory with three folders – one containing the train set, the validation set, and test set respectively. This report contains four main files:
-	Modules: Containing the VQVAE and PixelCNN models
-	Dataset: Classes for loading and preprocessing data
-	Train: Functions for training and saving the models, as well as plotting the losses.
-	Predict: Showing example usage of the trained models.
The ‘train’ and ‘predict’ files in particular have config specific variables (such as paths, embedding_dim etc.) which must be changed at the top of the file.
Any changes to the model and data can be made in Modules and Dataset respectively. To train the models, use the train.py file. Replace the current path variable with the local path to the OASIS dataset, and the names of the folders in the ‘DataPreparer’ instance, as well as the path where the models and losses will be saved. (Note that the training functions save the models as a new file every 20 epochs). The test functions in train.py also have optional parameters to visualise the data.
New images can be generated from the predict.py file. The function show_reconstructed_imgs(num_shown=2) shows the VQVAE input images, the codebook indices, and the reconstructions.
show_generated_indices(shown_imgs=2) shows the codebook indices generated by the PixelCNN. This calls cnn_model.sample() which takes in codebook indices, and generates any which have been replaced with ‘-1’. The line ‘masked_indices[:,:,16:,:] = -1’ can be changed to alter which indices are generated.
show_generated_output() shows a new generated image. Again, the masked_indices variable can be changed, so that parts of images can be generated based on existing pixels.

**Data Processing:**
 	
In the dataset class, the data is loaded into datasets. The transforms applied to all data is conversion to a tensor and normalisation, based on the mean and standard deviation of the entire dataset. The number of images in the train, validation, and test dataset was of the approximate ratio 20:2:1.

**Models Overview:**

Note – the VQVAE model described contains 32 embedding dimensions. This was the original model, but this has since been increased.
VQVAE
The VQVAE model can be broken up into the encoder, vector quantiser, and decoder. Throughout the process, the data is generally in the shape (Batch, Channel, Height, Width). The batch size used is 128, as this was found to yield the best results overall. A visual depiction and descriptions of the VQ-VAE components are below.
 ![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/0203148a-b794-4d82-862f-25fcaaa06baa)

(Image from Neural Discrete Representation Learning2, 2018)
Encoder:
(B=128, C=1, H=256, W=256) -> (B=128, C=32, H=32, W=32)
The encoder used is a convolutional neural network which maps the grayscale (single channel) input images into a continuous latent space with 32 channels. Three convolutions are performed, and after each convolution, the data is normalised, and a non-linear activation function is applied. Each convolution has a stride of 2, and height and width in the latent space is reduced by a factor of 8.
In the current model used, the encoder changes the shape of the data from (128, 1, 256, 256) to (128, 32, 32, 32). To modify any layers of the model, changes can be made to the self.encoder attribute. (Note that the channels of the encoder and decoder should correspond to the embedding dimension of the vector quantiser).
Vector Quantiser:
(B=128, C=32, H=32, W=32) -> (B=128, C=32, H=32, W=32)
The encoder contains an embedded space/codebook (which it learns). For each pixel, a vector of the 32 channels associated with it is compared to each of the 32-dimensional embeddings in the embedded space (codebook). The index of the embedding which is closest (based on Euclidean distance) to the given vector is returned. The embedding vector which corresponds to this index is then found, and thus, the quantiser maps the 32-channel continuous space to a discrete space.
This process is broken into a number of functions, increasing the readability of the code and allowing for these functions to be called individually when generating images. The functions include:
-	get_encoding_indices, which takes the original input (B, C, H, W), and saves the Euclidean distances to each embedding. It then finds the indices to the closest embedding, returning the codebook indices in the shape (B, H*W)
-	output_from_indices, which takes the codebook indices of shape (B, H*W), and returns the corresponding embedding vector. It also reshapes the output into the correct shape (B, C, H, W) before returning it.
The forward function calls both these functions and contains a straight-through gradient estimator (shown by the red arrow in the diagram above). This is because the discretisation is non-differentiable, and therefore back-propagation could otherwise not occur. If x is the input to the quantiser and Z(x), the output. The output x + Z(x) - x is still Z(x), however detaching Z(x) and -x in ensures that the backpropagation can follow the dotted line shown below, avoiding the non-differentiable Z(x) block.
 ![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/cb007ed1-244d-4931-b0e5-ab9a03007541)

The forward function also returns the codebook indices (reshaped back to B, H, W), as well as the associated loss – comprised codebook loss and the commitment loss. The codebook corresponds to the distance between the latent vectors produced by the encoder and the closest vectors in the codebook, which ensures that the vectors in the embedded space are effective representations. The commitment loss prevents the codebook vectors from oscillating between values, encouraging them to converge. The vector quantiser has a ‘beta’ attribute (currently 0.2), and the quaniser loss is calculated as quant_loss = codebook_loss + beta*commitment_loss. The ‘beta’ value can be changed to affect the robustness and aggression of the quantiser.
Decoder:
(B=128, C=32, H=32, W=32) -> (B=128, C=1, H=256, W=256)
The decoder takes the output of the vector quantiser B, C, H, W (currently B, 32, 32, 32), and performs three transposed convolutions to attempt to reconstruct the original image. Each transposed convolution is again followed by normalisation and a non-linear activation function.

The forward call of the VQVAE passes the image through the encoder, vector quantiser, then decoder. It calculates the reconstruction loss, which represents the dissimilarity between the input image and the output image. It returns the output image, vector quantiser loss, reconstruction loss, and the encoding indices (helpful for visualising). The VQVAE model also contains the function img_from_indices. This function is useful as it takes in the codebook indices (B, H, W) and allows for the constructed image to be returned directly, which is used during image generation.
PixelCNN
The autoregressive PixelCNN used is based on an implementation for MNIST data by Phillip Lippe1.
The pixel CNN attempts to learn the space of discrete codebook indices of the VQVAE, such that new plausible combinations can be predicted. This is achieved through a masked convolution, which does not allow the neural network to see the current or future pixels. This was achieved by creating a mask as shown below and multiplying with the kernel weight data before convolution.
<img width="112" alt="image" src="https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/40a82168-d0b4-4d59-b992-b3d69a7cc459">


However, using this kernel directly led to a blind spot in the receptive field, and therefore a gated convolution will be used, with both a vertical stack and horizontal stack, with architecture:
![image](https://github.com/DruCallaghan/PatternAnalysis-2023/assets/141201090/99deeeff-ca6d-48b7-b379-9a27b8f27944)
 
(Figure by Aaron van den Oord et al.):
The neural network then learns to predict future indices. A number of classes were used in developing the PixelCNN:
-	MaskedConvolution: This is the base class which takes a mask as a parameter and performs a masked convolution.
-	VerticalConv: This extends MaskedConvolution and is a vertical stack convolution which considers all pixels above the current pixel.
-	HorizontalConv: This extends MaskedConvolution and is a horizontal stack convolution which considers all pixels to the left of the current pixel. It also contains residual blocks, because the output of the horizontal convolution is used in prediction.
-	GatedMaskedConv: This performs the horizontal and vertical stack convolutions, and combines them according to the graph above.
The PixelCNN scales all of the indices to a range between 0 and 1. It performs the initial convolutions (which mask the centre pixel), and then a number of gated-masked convolutions. It applies a non-linear activation function, and reshapes the output to (Batch, Classes, Channels, Height, Width). This allows for the Cross-entropy loss between the actual indices and the generated indices to be found. Note that for this model, the loss is given by the bpd (bits per dimension), which is calculated based on the cross-entropy, but is more suitable for the model.
The model also contains a function called ‘sample’. This takes an input of indices, with any indices to generate replaced with ‘-1’. It iterates through the pixels and channels and uses Softmax to predict which indices are likely. For full image generation, the indices can be replaced entirely with ‘-1’s.

**References**

1.	Lippe, P. (no date) Tutorial 12: Autoregressive Image modelling, Tutorial 12: Autoregressive Image Modelling - UvA DL Notebooks v1.2 documentation. Available at: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html (Accessed: 09 October 2023)
2.	Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (30/05/2018), Neural Discrete Representation Learning.

