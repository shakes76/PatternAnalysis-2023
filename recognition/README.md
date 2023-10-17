# Recognition Tasks 5
# **Brain MRI Super-Resolution using ESPCN**
### **Description:**
Efficient Sub-Pixel Convolutional Neural Network (ESPCN) is designed to upscale low-resolution images to high-resolution counterparts. In this implementation, we aim to solve the problem of enhancing the resolution of Brain MRI scans. Given a down-sampled MRI image, our model attempts to produce a clearer image with a resolution upscale factor of 4.

### **How it Works:**
The ESPCN model initially extracts feature representations from low-resolution images. These features are then upscaled to the desired resolution using an efficient sub-pixel convolution layer. This method avoids the computationally expensive transposed convolution operation. The upscaled features are aggregated to produce a high-resolution output.

### **Dependencies:**
- Python 3.10
- torch 2.0.1+cu117
- torchaudio 2.0.2+cu117
- torchvision 0.15.2
- matplotlib 3.7.1
- numpy 1.25.0
- PIL 9.5.0


