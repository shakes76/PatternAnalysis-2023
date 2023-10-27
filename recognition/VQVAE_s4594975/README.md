# VQVAE and Pixel CNN - Generative Model of OASIS Dataset
AIM: The aim of this project is to develop a generative model for OASIS brain dataset using a VQVAE (Vector Quantized Variational Autoencoder) and a PixelCNN (Pixel Convolutional Neural Network) to produce "reasonably clear" medical images with a Structured Similarity Index (SSIM) score exceeding 0.6. The project involves leveraging advanced deep learning techniques to create high-quality, structured, and interpretable medical images, which can be invaluable for medical diagnosis, research, and analysis.


## OASIS DATASET

The OASIS dataset, which stands for "Open Access Series of Imaging Studies," is a collection of neuroimaging and clinical data designed for research in neurodegenerative diseases, particularly Alzheimer's disease. It is a valuable resource for researchers, clinicians, and scientists interested in studying brain health, dementia, and related conditions.

Traning Images - 9,664 
Test Images - 544
Validation Images - 1120

## VQVAE
A VQVAE, or Vector Quantized Variational Autoencoder, is a type of neural network architecture used in the field of deep learning and generative modeling. The VQ-VAE operates using a discrete latent space, which is represented as a discrete codebook. The encoder part of the model characterizes this latent space as a categorical distribution. The codebook is established by converting the continuous embeddings and the encoded outputs into discrete code words. These discrete code words are subsequently fed into the decoder. The decoder is then trained to produce reconstructed samples based on these discrete code words. 

VQVAE -> 3 parts. Encoder, latent space and decoder.

!Architecture picture

A VQVAE (Vector Quantized Variational Autoencoder) differs from a VAE (Variational Autoencoder) primarily in the nature of their latent space representations. In a VAE, the latent space is continuous and probabilistic, allowing for smooth data generation with continuous variations, while a VQVAE utilizes a discrete and quantized latent space achieved by mapping the continuous latent space into discrete codes.

## PIXEL CNN
PixelCNN is a type of generative model designed for generating images, particularly pixel by pixel. It models the conditional distribution of each pixel in an image given the previous pixels. It's capable of generating high-quality, highly structured images and can be used for various image generation tasks.

Pixelcnn -> Pixel Layer and Resolution block

VQVAE runs alongside pixelcnn which trains to generate encodings. After the training, the pixelcnn is used to generate encodings that aren't exposed to the VQVAE. These are then decoded by the decoder layer in the VQVAE.

## DATASET - PREPROCESSING
Preprocessed dataset was used which contained train, test and validation images. The dataset was normalised to be in the range [-0.5 to 0.5] and pixel size was downsampled to 80x80 from 256x256 to compile faster and less memory resources.

!size of the dataset

## TRAINING 
Training the models were executed on a machine with macOS software containing an intel core i5 2015 version. VQVAE and Pixel CNN were trained with 100 epocs each. Every epoch(76 steps) on the VQVAE took about 60s and every epoch(3 steps) on the Pixel CNN took about 4s.
They had a batch size of 128.

### VQVAE


VQVAE Loss Graph 

![VQVAE training loss 100 epoc](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/b1495d46-92b2-43e4-aa3f-a194cfc3c5e1)

### PIXEL CNN
Validation split - 0.7:0.3

Training Loss vs Testing loss

![Test vs train pcnn](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/b331ef75-0079-4309-bcaa-3c45070116f0)

Accuracy of the Pixel CNN

![Pcnn accuracy](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/ccf9d917-d753-46ba-a339-22b586828550)

## RESULTS
### VQVAE

Here are the test images of the VQVAE -> average structured similarity of 0.82 and are reasonably clear.

<img width="375" alt="Average ssim" src="https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/2ce96fce-d03e-446f-b3e1-8683cf4d3b46">

Random 10 images for testing VQVAE

![Figure_1](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/d0d43c65-6f78-4680-8e96-f9ea1ac6b6b6)

![2](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/460b3598-af32-47bf-919a-03e04d9a4d0e)

![3](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/b5f4814b-2a80-46b0-829b-ba410521265c)

![4](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/e7bc254c-7afd-444a-bb4e-0e7103d01523)

![5](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/6f70fb9b-4bcc-49af-bb6c-66545cb1fa12)

![6](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/58031ca0-34b9-4232-bbd9-6a15e1aa5ced)

![7](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/c9a97f20-5156-4eb9-8696-8372b4caf32b)

![8](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/5e1e4383-57fa-4d06-945b-7a27ab1d0394)

![9](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/8ac91c47-de9d-45f1-9ac0-2fbe7fd9f549)

![10](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/ae6477aa-5924-48ca-8eac-0c120fb91dfe)

### Pixel CNN

Genearted Codes by PCNN - Latent Vectorization Codebook samples. These were run through the encoder, hence are pixelated. 

![1](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/bdc37284-1782-4ab3-887e-994080fff8cb)

![2](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/62e9177b-d7a2-455f-9402-ca6739ae025c)

![3](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/6eb07507-a935-4aa1-bebe-51b6588caa8f)

![4](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/ac538113-2de7-47e1-8ef3-7382a035dfd4)

![5](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/fb808da8-06bd-4488-bbe2-3a1cc0c6b6b5)

![6](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/2331077b-9838-4487-827d-d68d1486162c)

![7](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/3d504c90-2008-4116-be82-be9bdb4ca7f1)

![8](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/c5c98b81-a2f3-49d4-afa1-c2948840e17d)

![9](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/bd14d720-9bcc-433e-b507-7789c33fad6c)

![10](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/bdee8415-09d3-46ec-b143-deba1529a845)

Generated Images - Feeding the Pixelated samples generated by the Pixel CNN to the VQVAE's decoder. They hadnt been exposed to the VQVAE model before. 


![Figure_1](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/dffcf712-538d-4154-a859-42f693c25ef2)

![2](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/9c2e7470-2e95-42a5-be51-9c385e30fdb6)

![3](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/4e72a961-0cef-48a7-a032-0a77ae57887d)

![4](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/24d680eb-fc94-4c33-abdf-af246f43ed3c)

![5](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/140938e5-cd78-463b-ab41-63eb87dd8caa)

![6](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/a6c83d3e-0af6-4a5c-bd8e-a41f7693f1b7)

![7](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/bbde8c1b-882c-43b9-a741-c61346f9445d)

![8](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/d9c5f73c-f418-4cfd-bc2f-583ddb9f5ab1)

![9](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/7d850fde-c1ed-470c-b7e9-33bebd3100ef)

![10](https://github.com/Xodacan/PatternAnalysis-2023/assets/88042327/1a40e21c-cc34-43b7-ba8b-29b6ceb0607b)



