# Generative Model of OASIS Brain Data Using VQ-VAE & PixelCNN
## Overview
This project constructs a generative model that can generate OASIS brain images using the VQ-VAE (Vector-Quantized Variational Autoencoders) model and PixelCNN model. First, use VQ-VAE to learn the OASIS dataset for optimizing encoding embeddings. Then, use the trained encoder to generate a codebook for training the PixelCNN model. Finally, randomly generate a new encoding matrix through PixelCNN, map it through the codebook, and decode it to obtain a new image.
## VQ-VAE Model
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Principle.png" width="">
Unlike variational encoders and autoencoders, the latent space in VQ-VAE is discrete rather than continuous, which can prevent posterior collapse. Additionally, the prior probability distribution in VQ-VAE is not static and can be learned through training. The encoder uses a convolutional network to generate corresponding features, then calculates the Euclidean distance, and maps the vectors to a discrete codebook. The decoder retrieves the corresponding code from the codebook based on the most recent embedding and uses the code words for data generation. 
  
### Gradient design
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Gradient.png" width="500">
The backpropagation of VAE is challenging because the process of obtaining the nearest code word during the forward pass is non-differentiable. The paper uses a straight-through estimator because the encoder and decoder have the same dimensions. By copying gradients and adjusting the direction of the corresponding encoding vectors, the encoder's output is continuously moved closer to the nearest code.
  
### Loss function
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Loss.png" width="500">
The loss function consists of three parts. 
1. Reconstruction loss: The difference between the encoder's input and the generator's output.
2. Loss for optimizing the encoding embedding: The code continuously moves closer to the input, learning the embedding.
3. Loss for optimizing the encoder's output, approaching the code in the codebook.

### Architecture
Due to the use of full-sized grayscale images, the input shape of the encoder is (n, 256, 256, 1). For the vector quantizer layer, I use 32 embeddings with a dimension of 128. 
<image src="https://github.com/MandalorianForce/PatternAnalysis-2023/blob/b6f7844532b9a8bc93d05abd04c8f08bef97ce4c/recognition/VQ-VAE-46495408/Images/VQ-VAE_Structure.png" width="500">

### Data preprocessing
The OASIS dataset I'm using has training, validation, and test samples already organized into separate folders. There are 9,664 samples for training, 1,120 for validation, and 544 for testing. I set the batch size to 128. Then standardized the images by dividing the pixel intensity values by 255, which rescales the data to a range of [0, 1], ensuring uniform distribution among all the images.

### Results
I trained for a total of 50 epochs, and around the 20th epoch, it was nearing convergence, at which point the SSIM reached 0.9.
<image src="" width="">
<image src="" width="">

Through encoding and reconstruction, SSIM achieved an average of 0.6 or higher on the test data.
<image src="" width="">

## Pixel CNN Model
### Architecture
<image src="" width="">

### Results
<image src="" width="">

## Future Outlook
The training results of VQVAE are quite noticeable, and the generated photos are relatively high-quality. However, the photos generated using PixelCNN have significant differences from real ones. I suspect this is a result of suboptimal training. There are several possible remedies to try:
1. Convergence is slow with each epoch, so you can try increasing the learning rate of the optimizer.
2. Increase the number of training epochs to improve the weight accuracy through extensive training.
3. Change the size of the input photos. Here, I used full-sized 256 * 256 photos, which are relatively large. You can compress them to speed up training.

## Dependencies
* `numpy` - 1.26.0
* `tesnorflow` - 2.14.0
* `tesorflow-probability` - 0.22.0
* `matplotlib` - 3.7.2

## References
* Oord, A. van den, Vinyals, O., &amp; Kavukcuoglu, K. (2018a, May 30). Neural Discrete Representation Learning. arXiv.org. https://arxiv.org/abs/1711.00937v2 
* Team, K. (n.d.). Keras documentation: Vector-quantized variational autoencoders. https://keras.io/examples/generative/vq_vae/ 
* Team, K. (n.d.-a). Keras Documentation: Pixelcnn. https://keras.io/examples/generative/pixelcnn/ 
