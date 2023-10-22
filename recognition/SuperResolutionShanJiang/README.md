# Brain MRI super-resolution network 
## Introduction
This project implemented a super-resolution CNN network trained on the ADNI brain dataset. The trained model can convert a low resolution image into a high resolution version. The CNN consists of four convolutional layers followed by a depth-to-space transformation. The dataset used in training and testing is ADNI dataset, including 2D slices of MRI for both Alzheimer’s disease patients (AD) and health control (HC). For our purpose, dataset for AD and HC are combined together into one dataset (since our model do not deal with classification). The training and testing achives PSNR of 28.82 and ? WITH loss of 0.0013 and? respectively.
## Getting Started
### Install the required dependencies
pip install -r requirements.txt
### Loading dataset
The dataset used for training(and validation) and testing is loaded in `dataset.py`.The images are cropped into specified size (256*248 in our case). 20% of the training dataset is reserved for validation. Pixel values of traning and validation images are rescales to range of 0 to 1. A list of path for each testing image is also created for later use.        
Then we produce pairs of  high resolution and loss resolution images from the training and validation dataset. To get high resolution images,we convert images from the RGB color space to the YUV colour space and only keeps Y channel. To get low recolution version, we convert images from the RGB color space to the YUV colour space,only keeps Y channel and resize them by certain ratio (4 in our case) so that their resolution is reduced. Each pair is put into a tuple to be fed into the model for training. The following images show exmaple of a high resolution image and the corresponding low resolution image. 
![A high resolution MRI image](readme_images/high_res_train.png)
![A low resolution MRI image](readme_images/low_res_train.png)
To run dataset.py, follow following steps:
1. Define the exact directory containing training dataset(directly contains image files) at line ?? by altering the value of variable `data_dir`.
2. Define the exact directory containing testing dataset(directly contains image files) at line ?? by altering the value of variable `test_path`.
3. Define the exact directory containing images to be predicted on(directly contains image files) at line ?? by altering the value of variable `prediction_path`. Those images is used to provide a demo of the prediction result of the model in `predict.py`
4. (optional) change the value of `upscale_factor` at line ?? to downsample your training and validation images by a different ratio
5. Change the values of `crop_width_size` at line ?? and `crop_height_size` in line ?? to make sure they are less than or equal to the orginal width and height of the images, and is divisible by `upscale_factor`.
### Building model
The model structure is defined in `modules.py`. using keras framwork. The structure of the model is as following:
- first layer: A convolutional layer with 64 filters and a kernel size of 5 to extract features.
- second layer: A convolutional layer with 64 filters and a kernel size of 3 to extract features.
- third layer: A convolutional layer with 32 filters and a kernel size of 3 to extract features.
- fourth layer: A convolution layer with `channels * (upscale_factor ** 2)` filters and a kernel size of 3 to increase spatial resolution.
- depth to space operation: Using TensorFlow's tf.nn.depth_to_space function to perform a depth-to-space upscaling operation specified 'upscale_factor' to produce the super-resolved image with a higher resolution.
Note: for best performance, keep the value of `upscale_factor` parameter (default to 4) in `get_model` the same as the value of `upscale_factor` parameter defined in `dataset.py` and keep the value of `channels` to the default value (1).
### Utilities
Two functions are defined in `utils.py`. 
- `get_lowres_image(img, upscale_factor)` downsamples given `image` by  ratio of given `upscale_factor`. It is later used in train.py to convert testing images to low resolutions images.
- `upscale_image(model, img)` preprocessed given `image` and use the give `model` to increase its resolution. The preprocessing include convert the image into YCbCr color space and isolate and nomalise(dividing by 255) the Y channel, reshape the Y channel array to shape shape matches the model input shape. The prediction output from the model is demornalised(multiplying by 255) and restored to RGB color space.


       

     
    
