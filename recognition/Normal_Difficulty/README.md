# brain MRI super-resolution network


## Introduction of the problem

The aim of this model is to provide a high resolution version of the input brain MRI images. The model used in the task is called ESPCN (Efficient Sub-Pixel CNN), is a model that reconstructs a high-resolution version of an image given a low-resolution version. It involves using convolutional neural networks (CNNs) and specifically a sub-pixel convolution layer to upscale low-resolution images to a higher resolution. 

## dataset
The dataset comprises 21,520 brain MRI images and is categorized into AD (Alzheimer's Disease) and NC (Normal Control). The dataset is divided into two subsets: the training dataset, which encompasses 80% of the images, and the validation dataset, which encompasses the remaining 20% of the images. Test data is also loaded. To run this file, `relative_train_path` and `relative_test_path` will need to be altered to fit the data folder path.

## machine learning model
The model is a convolutional neural network (CNN) designed for super-resolution tasks. It takes an input image of variable dimensions and channels. The network consists of four convolutional layers with 64, 64, 32 filters and a dynamic number of filters which is determined by upscale factor . After feature extraction, the `depth_to_space` function rearranges the feature maps to achieve an upscale factor of `upscale_factor`. This architectural choice is vital for super-resolution, enabling the model to increase the image's spatial resolution. 

## utilities
The untils.py file contains two function, `get_lowres_image` and `upscale_image`. The frist funcion take input of a image and reduces resolution, the second funtion took the low resolution image to fitin the model and increase resolution.

## training and testing
The training process includes various callbacks for monitoring and early stopping, as well as model checkpointing. 
`ESPCNCallback`: A custom Keras callback that calculates and prints the Peak Signal-to-Noise Ratio (PSNR) during training. It stores PSNR values for each epoch.
`keras.callbacks.EarlyStopping`: Monitors training loss and stops training if it doesn't improve for a specified number of epochs (patience).
`keras.callbacks.ModelCheckpoint`: Saves the model's weights during training, allowing you to keep the best-performing model.


## prediction results