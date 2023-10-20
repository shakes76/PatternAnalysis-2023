# Brain MRI Super-Resolution Network
This project aims to improve the resolution of MRI brain scans. This is done through the use of an efficient sub-pixel convolutional neural network (ESPCN) to upscale image resolution by a factor of 4.

## How it works:
![Model Structure of ESPCN.](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*AnTunkGkz-KNTQkrezoSmQ.jpeg)
The model takes low-resolution images, created by downsampling from high resolution images and outputs reconstructed images upscaled by a factor of 4. Structure can be seen as above; it extracts a number of feature maps through a series of convolutional layers. It then makes use of an efficient sub-pixel convolutional layer to reconstruct a higher resolution image. The sub-pixel convolution works by performing a convolution to combine the various sub-pixels of different feature maps. The system makes use of MSE to calculate the loss between reconstructed images and ground truth higher resolution images.

## Visualisation of results:
### Comparison of Images:
![comparison](https://github.com/zharbutt/PatternAnalysis-2023/assets/141378636/5c3ffd30-82ea-44be-8c3b-d38ec60d860d)

### Loss visualisation:
![training_loss](https://github.com/zharbutt/PatternAnalysis-2023/assets/141378636/536b0d45-c34d-4e1d-a4ce-24044f32dd0d)

## Dependencies
Python 3.10  
pytorch 2.01  
pytorch-cuda 11.7  
torchvision 0.15.2  
matplotlib 3.7.2  
numpy 1.25.2  
scikit-image 0.20.0  
Pillow 9.4.0  

## References
[Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/#image-superresolution-using-an-efficient-subpixel-cnn)  
[ADNI Data](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI)
