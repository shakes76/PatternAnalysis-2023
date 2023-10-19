# Recognition Tasks 5
# **Brain MRI Super-Resolution using ESPCN**
Student Information  
Name: Zijun Zhu  
ID: s4627546  
### **Description:**
Efficient Sub-Pixel Convolutional Neural Network (ESPCN) is designed to upscale low-resolution images to high-resolution counterparts. In this implementation, we aim to solve the problem of enhancing the resolution of Brain MRI scans. Given a down-sampled MRI image, our model attempts to produce a clearer image with a resolution upscale factor of 4.

### **How it Works:**
The ESPCN model initially extracts feature representations from low-resolution images. These features are then upscaled to the desired resolution using an efficient sub-pixel convolution layer. This method avoids the computationally expensive transposed convolution operation. The upscaled features are aggregated to produce a high-resolution output.

1. **Initial Convolution**: The input image, a grayscale MRI scan, is passed through a convolutional layer with 64 filters and a kernel size of 5x5. This layer captures the primary features of the image.
2. **Intermediate Convolution Layers**: The output from the initial convolution is processed by two more convolutional layers, both having 64 filters. The first of these uses a 3x3 kernel, while the second uses a 3x3 kernel but reduces the output channels to 32. These layers further refine the features extracted from the image.
3. **Sub-Pixel Convolution Layer**: This layer increases the number of channels in the image by a factor equivalent to the desired upscale factor squared. For a 4x upscale, the number of channels is increased 16-fold.
4. **Pixel Shuffle**: The increased channels are then reshuffled to form a higher resolution image using the pixel shuffle operation.
The combination of these operations allows the ESPCN model to effectively enhance the resolution of MRI scans, ensuring that the upscaled image retains the necessary details and patterns present in the original scan.

![model_architecture](https://github.com/a12a12a12a12/PatternAnalysis-2023/assets/90440194/b1e3b5da-b561-4e7e-9130-30e35d5b7411)

### **Usage/Performance:**
#### **Dataset**
[ADNI brain dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI) (Blackboard on COMP3710-Course Help/Resources
)  

├── project-root/AD_NC  
│   ├── train  
│   │   ├── AD  
│   │   └── NC  
│   ├── test  
│   │   ├── AD  
│   │   └── NC  




##### **input**
A low-resolution MRI scan of dimensions 64x60.
Here, we only need to run predict.py 

```python
# NOTE: set the model dir here!
model.load_state_dict(torch.load('best_model_v2.pth'))
```
##### **output**
An upscaled MRI scan of dimensions 256x240.
#### **performance**
![result](https://github.com/a12a12a12a12/PatternAnalysis-2023/assets/90440194/4d09c59a-df61-4031-90df-3fc1d8d8820f)


### **Pre-processing and training**
- The MRI images were first normalized to a range between -1 and 1 to ensure stability during training. 
- use images in AN_NC/train folder to train，The dataset was split into training and validation sets in an 8:2 ratio. 
  ![ret](https://github.com/a12a12a12a12/PatternAnalysis-2023/assets/90440194/2f8b9160-c949-4d0b-8cfe-ccf815a444cf)

### **Dependencies:**
- Python 3.10
- torch 2.0.1+cu117
- torchaudio 2.0.2+cu117
- torchvision 0.15.2
- matplotlib 3.7.1
- numpy 1.25.0
- PIL 9.5.0

### **Reference:**
- [ADNI brain dataset](https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI) (Blackboard on COMP3710-Course Help/Resources)  
- [Efficient Sub-Pixel CNN]([https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI](https://keras.io/examples/vision/super_resolution_sub_pixel/)https://keras.io/examples/vision/super_resolution_sub_pixel/) 
