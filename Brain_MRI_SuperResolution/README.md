# PatternAnalysis-2023
## Report - Super Resolution CNN Network
**Author**: Abhinav-48244178

Super resolution cnn network:

Report details the implementation of a Brain MRI super-resolution network trained on ADNI dataset.This approach leverages a super-resolution CNN layer to generate high-resolution images from their lower-resolution counterparts. By comprehending patterns and structures within the downscaled training images, the network applies this
knowledge to upscale the images effectively. In Today time, the significance of such super-resolution techniques are implemented over large scale either it be medical field or satellite imaging almost everything requires visualization.
         
Description of Algorithm:
                   
The Super-Resolution Convolutional Neural Network (SRCNN) is used for single image super resolution (SR).
Network operates in distinct phases to upscale images.
Firstly, through a process of patch extraction and representation, it extracts patches from the low-resolution images.
Then, during the non-linear mapping phase, the network harnesses its capacity to learn intricate patterns and details
from the data. Lastly, in the reconstruction phase, the final layer of the SRCNN is dedicated to generating the high-resolution image from the acquired feature maps.The filters within this layer are meticulously trained to collate the information from these feature maps, ensuring the output is a finely upscaled image.               
SRCNN models have various implementations across different domains. They offer a more feasible approach to obtaining high-resolution images than relying on the availability of high-cost sensors and advanced optics manufacturing technology. The medical field greatly benefits from SRCNN due to its ability to enhance image quality, which is pivotal for precise diagnostics and treatments.  

Refer to image "Brain_MRI_SuperResolution/Images/model.png"                                                               


Working of Resolution Network :

On a large dataset. First,on running data_loader preprocessing of the the images, which involves padding and down sampling resulting in creation of new folders. The adjusted images are fed into the Sub-pixel CNN model. Inside the model, a residual block ensures making training easier and tackles the vanishing gradient problem, allowing for smoother training. The model then increases the detail of these images using a traditional convolutional layer and further enhances it with a sub-pixel convolution layer. Finally, a deconvolution layer produces the high-resolution image.
For training,mixed preprocessed samples from different categories are divided into training and validation groups. During training, model uses a PSNR metric, the Adam optimizer, and try to minimize the mean squared error. The best version of the model is saved based on validation results and stop training if no progress is seen after several attempts. Once training is done, we use charts to show how training went.
For predictions, a function called displayPredictions takes a test image, resizes it, and makes a low-quality version of it.Both of the original and low-quality image are displayed. Then, using our trained model, upgradation the low-quality image to high-resolution takes place. The model is loaded from a specific place on our computer, and we finish by showing how good the model is with a test image.

refer to image (Images/SRCNN.png)

Problem it solves :

SRCNN makes the enhancing of old photographs or videos very easy. This comes in handy especially when trying to improve the quality of CCTV footages or upscaling digital media to be compatible with larger screens. 
also particularly useful application is the restoration of images that have undergone heavy compression. Such images often lose significant detail and exhibit compression artifacts; SRCNN can mitigate these effects by restoring some of the lost details. In cases where images contain text, conventional upscaling methods often blur or render the text illegible. SRCNN, however, can produce clearer and more readable text upon upscaling. In specialized sectors, such as geospatial analysis and medical imaging, the need for high-resolution images is paramount. SRCNN can play a crucial role here by enhancing satellite and aerial imagery or providing clearer views from MRI and CT scans.


Data Splitting:
 
90% Training data:
The majority of the data is used for training the model. Using a larger portion of data for training ensures that the model has a sufficiently diverse set of samples to learn from, which helps in achieving better model generalization.
           
10% Validation data:
It's used to evaluate the model's performance on unseen data after every epoch, which helps in monitoring overfitting. EarlyStopping is used in conjunction with the validation loss to halt training if the model starts to overfit (i.e., if the validation loss does not improve for a set number of epochs).

for testing purpose variation between different resolution images is represented as figures.


Dependencies and Constants: 

1.Numpy
2.Tensorflow
3.Keras
4.Matplotlib
5.PIL (from the Image and ImageOps imports)

Constants.py 

All of the constants utilised by the Super-resolution CNN model.

WIDTH = 256
HEIGHT = 240
DOWNSCALE_FACTOR = 4
SEED = 1
RESIZE_METHOD = "BILINEAR"
EPOCHS = 15



Inputs: 

Input1 - (Input_output_samples/input1.png)
Input2 - (Input_output_samples/input2.png)


Outputs: 

Output1 - (Input_output_samples/Output1.png)
Output1 - (Input_output_samples/Output1gray.png)
Output2 - (Input_output_samples/Output2.png)
Output2 - (Input_output_samples/Output2gray.png)
PSNR - (Images/PSNR.png)
Training vs validation - (Images/training_plot.png)


References:

https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c
https://cs229.stanford.edu/proj2020spr/report/Garber_Grossman_Johnson-Yu.pdf
https://keras.io/examples/vision/super_resolution_sub_pixel/
