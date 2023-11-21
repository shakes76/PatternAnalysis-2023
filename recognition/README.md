# Project Explain

## Problem We solved in this project

In this project, we will use the build a generative model using vq-vae and pixelCNN. The model needs to generate clear images and the SSIM should be over 0.6.


## Operating Process

1.	run the train.py model first, which will automatically save recon images and model to the directory, 
2.	run the PixelCNN.py, this file will use the trained vq-vae model to form the code book, save the code book (model). 
3.	Finally, run the predict.py. This file should have the visualisation of reconstructed images (in animation of all the reconstruction process) and the PixelCNN generated images saved under the path “./Pic/PixelCNN_generate.jpg”.


## Details of data-preprocess, validation and training

In this project, the dataset is preprocess using Grayscale and resize. (resize can be deleted if GPU memory is enough). 
Inside vq-vae training part, the training loopwill run for 15000 times, mse_loss is used to track the loss; Adamw is used for optimizer; multistep learning rate is used to adjust to avoid overfitting problems.

In validation part, save_image and state_dict are used to save generated images and model into the correct directory. 


## Dependence 

In this project, the dependence we used include: (they are all the latest version)
Python3

Used to transform data as follow:
from torchvision import transforms, datasets
from torchvision.transforms import Normalize, Compose, ToTensor
from torch.utils.data import DataLoader 

Used for training as follow:
import torch, 
import torch.nn as nn, 
import torch.nn.functional as F, 
from torch.optim import Adam

Used for predicting as follow:
import os, 
import random, 
import time, 
from PIL import Image

Used for training as follow:
import matplotlib.pyplot as plt, 
from matplotlib.animation import FuncAnimation, 
from torch.optim.lr_scheduler import MultiStepLR, 
from skimage.metrics import structural_similarity as compute_ssim, 
from torchvision.utils import save_image, 
from tqdm import tqdm



## Coding Logic
In this VQ-VAE project, a VQ class, a decoder, an encoder with residual layer and residual block class is used in the model for training. 
VQ works like the picture shown in the following workflow. Pictures are sent to Ze(x) via CNN. Which forms an encoder; q(z|x) aimed to find Zq(x) by using argmin (Euclidean distance, also can use cosine distance in our coding Euclidean distance is used to make things easier), which can find the closest vector out of codebook. Then the Zq(x) is given to the same CNN (transposed) to generate the image the same size with the input images.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/12ed40ac-01f0-4973-98e4-4732acce94ad)
(VQ-VAE)

resource: https://arxiv.org/pdf/1711.00937.pdf [1]

In the above picture, input images are given to a Pixel CNN, which forms an encoder, the encoder with compress the size of the original image; the output images are also given through the same Pixel CNN (transpose) to make sure the input images have the same size with the output images. These steps are the same with VAE (Variational Auto-Encoding)

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/1f152a5a-a141-47b0-bac1-900900a258ae)
(VAE)

resource: http://kvfrans.com/content/images/2016/08/vae.jpg [2]

However, the difference between VAE and VQVAE lies in how to build latent vector. Inside VAE, mean and deviation are used to form the latent space and it is continuous, so it is derivatively. However, inside VQVAE the latent space is discrete, not derivatively. The mean and deviation are used to form a discrete codebook. 
	Inside VQ class, 3 parts are used to form the loss function: reconstruction loss (optimize encoder and decoder), take stop gradient (output from encoder) as a constant ->which let the embedding get closer and closer to the output, commitment loss (take embedding as a constant -> which let Ze(x) get closer and closer to embedding.


## Pixel CNN

Pixel CNN generally works by generating one pixel within one epoch, and use the trained pixel to the next one. Inside our code, Pixel CNN is divided into two types, “A” type basically works for only the first layer convolution, while “B” type works for other layer convolution except the first layer. Then, we will apply a residual block and mask layer for our project. 
Pixel CNN is built in this project to generate quantized x to form a code book, the code book will be used for decoder to generate images. 



## Evaluation Process
In the first generation, the images is colourful, because plt.imshow is used directly to show the image, the default output of matplotlib is RGB 3 channels, it is like pic1. To solve this problem, gray scale is used as follow: plt.imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1, interpolation='none'), then it changed into pic2.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/84702278-ecd6-4d21-92aa-08fc556e577a)

Pic1.									

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/11d1f8e2-8777-4971-be52-51d9cc006727)

Pic2.

Now, the generated picture is looks reasonably like a brain MRI image and SSIM reaches can reach 80%, but it is still blur. Deep into the training code, autocast() is used to improve the speed of training, but the generated picture is not clear enough. Because autocast() will lower the graphic precision although it will accelerate the training speed. In the training, the min loss of using autocast() training 15000 times is 0.006, however if autocast() is not used, the min loss training 15000 is 0.0015. Besides, the matplotlib may also lower the resolution, so save_image inside torchvision.utils is used to take place of imshow(), the 5000 times trained picture is generated like Pic3. 

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/da8a8eaa-1691-4f69-ae94-752b7ebf581c)

Pic3.

It is still blur, but the sharpness of the images is improved and the SSIM reaches 90%. Next, we only need to train more times, so when we train 20000 times, the output is shown in Pic4.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/5a680481-d186-4513-aa39-07a05730cdc4)

Pic 4.

In this training, it is clear enough and SSIM could reach 95% one picture of all these 64 brain pictures is not shown, which means it could be overfitting. Therefore, we have to adjust the learning rate and track the loss during the training process. Since the first 1000 times training have high loss, we only record the loss after training 1000 epochs. The result is as follows: Pic5 is when learning rate equals 0.01, Pic6 is when learning rate equals 0.001, Pic7 is when learning rate equals 0.0001. The best images they can generate is also shown in Pic 8, 9, 10.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/a090cf33-0e28-4e51-9d51-cabb5fc1c014) Pic5. (lr=0.01)	![image](https://github.com/Albert-bc/vq-vae/assets/59477394/6307499f-461a-4cb7-8a9b-0f4602c303de) Pic8. (No_4543 _Loss_0.00182_SSIM_92.36%)     

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/3ec97594-196f-4ddc-9bde-a1d2dc587580) Pic6. (lr=0.001)  ![image](https://github.com/Albert-bc/vq-vae/assets/59477394/dceaab87-88ec-4007-b563-aed41eec277e) Pic9. (No_6726 _Loss_0.00117_SSIM_97.073%)	  

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/126a93ce-776c-4b4b-adf7-d0c07770acbd) Pic7. (lr=0.0001)  ![image](https://github.com/Albert-bc/vq-vae/assets/59477394/ae1995b9-23a8-45ea-b83b-ce62e68418fd) Pic10. (No_9693 _Loss_0.001575_SSIM_95.94%)

From the images, we can clearly see that inside 10000 times training, SSIM and loss performance is the best when learning rate equals 0.001, but it still overfitting because 1/64 image is missing (all black). Therefore, we need to adjust the learning rate to make it change in multistep. Code: from torch.optim.lr_scheduler import MultiStepLR is used to adjust the learning rate. However, it is easily getting loss like Pic11. if the learning rate step is not chosen well. After several attempts, we get the best generated image so far is shown in Pic12.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/9ea36fec-8a6e-4cd7-85ea-7009d848f835)

Pic11.

![image](https://github.com/Albert-bc/vq-vae/assets/59477394/9fd24b25-cbf9-4380-a6a4-0f9ce211c5ad)

Pic12. (1024 x 1024 pixel) (No_14165_img_Loss_0.00107_SSIM_97.37%)

In Pic12., the loss decreases to 0.0010 and SSIM increases to 97.0%. The image now looks quite reasonable clear, and SSIM is quite high. If we continue want to improve the resolution of the generated image, we can delete the line: transforms.Resize((128, 128)). This line only works to make sure the code works functionally on low memory GPU. Then, we can get the output image as clear as the input image


Then, build the PixelCNN + decoder (trained before) to generate new images

After attempts, the best image that can be generated by PixelCNN is shown in Pic13. (It will take a long time to run, 10000 times is shown in the code but 20000 are tested on my own device)


![image](https://github.com/Albert-bc/vq-vae/assets/59477394/a58b72ff-08bd-4c5b-afd4-fe3f9f9a7894)

Pic13. (1024 x 1024 pixel) (No_20000_SSIM_97.37%)



## Reference:

[1] resource from https://arxiv.org/pdf/1711.00937.pdf

[2] resource from http://kvfrans.com/content/images/2016/08/vae.jpg

