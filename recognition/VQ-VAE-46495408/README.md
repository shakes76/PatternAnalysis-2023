# Generative Model of OASIS Brain Data Using VQ-VAE & PixelCNN
## Overview
This project constructs a generative model that can generate OASIS brain images using the VQ-VAE (Vector-Quantized Variational Autoencoders) model and PixelCNN model. First, use VQ-VAE to learn the OASIS dataset for optimizing encoding embeddings. Then, use the trained encoder to generate a codebook for training the PixelCNN model. Finally, randomly generate a new encoding matrix through PixelCNN, map it through the codebook, and decode it to obtain a new image.
## VQ-VAE Model
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Principle.png" width="">
Unlike variational encoders and autoencoders, the latent space in VQ-VAE is discrete rather than continuous, which can prevent posterior collapse. Additionally, the prior probability distribution in VQ-VAE is not static and can be learned through training. The encoder uses a convolutional network to generate corresponding features, then calculates the Euclidean distance, and maps the vectors to a discrete codebook. The decoder retrieves the corresponding code from the codebook based on the most recent embedding and uses the code words for data generation. 
  
### Gradient design
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Gradient.png" width="400">
The backpropagation of VAE is challenging because the process of obtaining the nearest code word during the forward pass is non-differentiable. The paper uses a straight-through estimator because the encoder and decoder have the same dimensions. By copying gradients and adjusting the direction of the corresponding encoding vectors, the encoder's output is continuously moved closer to the nearest code.
  
### Loss function
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Loss.png" width="400">
  
The loss function consists of three parts. 
1. Reconstruction loss: The difference between the encoder's input and the generator's output.
2. Loss for optimizing the encoding embedding: The code continuously moves closer to the input, learning the embedding.
3. Loss for optimizing the encoder's output, approaching the code in the codebook.

### Architecture
Due to the use of full-sized grayscale images, the input shape of the encoder is (n, 256, 256, 1). For the vector quantizer layer, I use 32 embeddings with a dimension of 128. 

<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Structure.png" width="300">

### Data preprocessing
The OASIS dataset I'm using has training, validation, and test samples already organized into separate folders. There are 9,664 samples for training, 1,120 for validation, and 544 for testing. I set the batch size to 128. Then standardized the images by dividing the pixel intensity values by 255, which rescales the data to a range of [0, 1], ensuring uniform distribution among all the images.

### Results
I trained for a total of 50 epochs, and around the 20th epoch, it was nearing convergence, at which point the SSIM reached 0.9.

<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/2d260f327f05eb556d3f7971ac45dad1023ca8b9/recognition/VQ-VAE-46495408/Images/vqvae_training_loss.png" width="300">

Through encoding and reconstruction, SSIM achieved an average of 0.6 or higher on the test data.
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/547967dd50082d3618473d6ab722719a44fecb63/recognition/VQ-VAE-46495408/Images/vqvae_test_images.png" width="500">

## Pixel CNN Model
PixelCNN is a type of deep neural network autoregressive model that captures the distribution of dependencies between pixels in its parameters. It generates one pixel at a time in an ordered sequence along two spatial dimensions in an image. Using convolutional operations, PixelCNN can simultaneously learn the distribution of all pixels in an image.

<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/547967dd50082d3618473d6ab722719a44fecb63/recognition/VQ-VAE-46495408/Images/PixelCNN_Principle.png" width="500">

### Architecture
I uses two residual block layers and two 2D convolutional layers.

<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/2d260f327f05eb556d3f7971ac45dad1023ca8b9/recognition/VQ-VAE-46495408/Images/PixelCNN_Structure.png" width="300">

### Results
The convergence is very slow when training at around 100 epochs, and the subsequent loss is also around 0.7, with an accuracy of around 0.75.

<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/547967dd50082d3618473d6ab722719a44fecb63/recognition/VQ-VAE-46495408/Images/pixelcnn_loss.png" width="300">
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/547967dd50082d3618473d6ab722719a44fecb63/recognition/VQ-VAE-46495408/Images/pixelcnn_accuracy.png" width="300">

The generated code and image are as follows:

<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/1294fa615793764af35dd1064c52a7181b9b90e5/recognition/VQ-VAE-46495408/Images/pixel_test_image_1.png" width="300">
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/1294fa615793764af35dd1064c52a7181b9b90e5/recognition/VQ-VAE-46495408/Images/pixel_test_image_2.png" width="300">
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/1294fa615793764af35dd1064c52a7181b9b90e5/recognition/VQ-VAE-46495408/Images/pixel_test_image_3.png" width="300">

## Future Outlook
The training results of VQ-VAE are quite noticeable, and the reconstructed photos are relatively high-quality. However, the photos generated using PixelCNN have significant differences from real ones. I suspect this is a result of suboptimal training. There are several possible remedies to try:
1. Convergence is slow with each epoch, so you can try increasing the learning rate of the optimizer.
2. Increase the number of training epochs to improve the weight accuracy through extensive training.
3. Change the size of the input photos. Here, I used full-sized 256 * 256 photos, which are relatively large. You can compress them to speed up training.

## Usage
* `dataset.py`: There are 3 functions using `tf.keras.utils.image_dataset_from_directory()` to load data from folders. Use `preveiw_images()` to inspect the dataset. You can also resize images, but remeber to change the input size of models.
* `train.py`: `train_vavae()` and `train_pixelcnn()` are training functions. For the results, use `plot_vqvae_history()` and `plot_pixelcnn_history()` functions to visualize. `visualize_vqvae_results()` generates reconstructed images using test dataset.
* `predict.py`: Generates priors and uses decoder to generate fake images according to the codebook.

## Dependencies
* `numpy` - 1.26.0
* `tesnorflow` - 2.14.0
* `tesorflow-probability` - 0.22.0
* `matplotlib` - 3.7.2

## References
* Oord, A. van den, Vinyals, O., &amp; Kavukcuoglu, K. (2018, May 30). Neural Discrete Representation Learning. arXiv.org. https://arxiv.org/abs/1711.00937v2
* Oord, A. van den, Kalchbrenner, N., &amp; Kavukcuoglu, K. (2016, August 19). Pixel recurrent neural networks. arXiv.org. https://arxiv.org/abs/1601.06759v3 
* Team, K. (n.d.). Keras documentation: Vector-quantized variational autoencoders. https://keras.io/examples/generative/vq_vae/ 
* Team, K. (n.d.-a). Keras Documentation: Pixelcnn. https://keras.io/examples/generative/pixelcnn/
