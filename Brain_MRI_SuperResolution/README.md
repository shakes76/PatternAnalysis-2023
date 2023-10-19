# PatternAnalysis-2023
Pattern Analysis of various datasets by COMP3710 students at the University of Queensland 


Super resolution cnn network:

Report details the implementation of a Brain MRI super-resolution network trained on images from the ADNI dataset. 
This approach leverages a super-resolution CNN layer to generate high-resolution images from their lower-resolution
counterparts. By comprehending patterns and structures within the downscaled training images, the network applies this
knowledge to upscale the images effectively. In Today time, the significance of such super-resolution techniques implemented over large scale either it be medical field or satellite imaging and number of domains.
         
Description of Algorithm:
                   
The Super-Resolution Convolutional Neural Network (SRCNN) is used for single image super resolution (SR).
Network operates in distinct phases to upscale images.
Firstly, through a process of patch extraction and representation, it extracts patches from the low-resolution images.
Then, during the non-linear mapping phase, the network harnesses its capacity to learn intricate patterns and details
from the data. Lastly, in the reconstruction phase, the final layer of the SRCNN is dedicated to generating the high-resolution image from the acquired feature maps.The filters within this layer are meticulously trained to collate the information from these feature maps, ensuring the output is a finely upscaled image.               
SRCNN models have various implementations across different domains. They offer a more feasible approach to obtaining high-resolution images than relying on the availability of high-cost sensors and advanced optics manufacturing technology. The medical field greatly benefits from SRCNN due to its ability to enhance image quality, which is pivotal for precise diagnostics and treatments.     

Refer to image "Brain_MRI_SuperResolution/Images/model.png"                                                               


Working of Resolution Network :

On a large dataset. First, preprocessing of the the images takes place, which involves padding and down sampling. The adjusted images are fed into the Sub-pixel CNN model. Inside the model, a residual block ensures making training easier and tackles the vanishing gradient problem, allowing for smoother training. The model then increases the detail of these images using a traditional convolutional layer and further enhances it with a sub-pixel convolution layer. Finally, a deconvolution layer produces the high-resolution image.
For training,mixed preprocessed samples from different categories are divided into training and validation groups. During training, model uses a PSNR metric, the Adam optimizer, and try to minimize the mean squared error. The best version of the model is saved based on validation results and stop training if no progress is seen after several attempts. Once training is done, we use charts to show how training went.
For predictions, a function called displayPredictions takes a test image, resizes it, and makes a low-quality version of it.Both of the original and low-quality image are displayed. Then, using our trained model, upgradation the low-quality image to high-resolution takes place. The model is loaded from a specific place on our computer, and we finish by showing how good the model is with a test image.

Data Splitting:
 
90% Training data:
The majority of the data is used for training the model. Using a larger portion of data for training ensures that the model has a sufficiently diverse set of samples to learn from, which helps in achieving better model generalization.
           
10% Validation data:
It's used to evaluate the model's performance on unseen data after every epoch, which helps in monitoring overfitting. EarlyStopping is used in conjunction with the validation loss to halt training if the model starts to overfit (i.e., if the validation loss does not improve for a set number of epochs).

for testing purpose variation between different resolution images is represented as figures.
