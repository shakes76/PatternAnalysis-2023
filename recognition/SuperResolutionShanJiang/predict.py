from utils import *
from modules import *
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from tensorflow import keras
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
import os
import math
import matplotlib.pyplot as plt


# Reference
""" Title: Image Super-Resolution using an Efficient Sub-Pixel CNN
Author: Xingyu Long
Date: 28/07/2020
Availability: https://keras.io/examples/vision/super_resolution_sub_pixel/"""

# load the trained model
checkpoint_filepath= "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/checkpoint/"
model = get_model()
model.load_weights(checkpoint_filepath)

# Specify path to store prediction result
prediction_result_path = "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/prediction_result/"


# Dowansample resolution of iamges by factor of 4, then predict higher resolution image using the model
total_bicubic_psnr = 0.0 # PSNR of downsampled image
total_test_psnr = 0.0 # PSNR of model output



for index, prediction_img_path in enumerate(get_prediction_img_paths()):
    img = load_img(prediction_img_path)
    lowres_input = get_lowres_image(img, upscale_factor) # downsample
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input) # Predict
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
    print("higher resolution")
    display(array_to_img(highres_img))
    print("lower resolution")
    display(array_to_img(lowres_img))
    print("prediction")
    display(array_to_img(prediction))
    # array_to_img(prediction).show()

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr
    
    image1 = array_to_img(highres_img)
    image2 = array_to_img(lowres_img)
    image3 = array_to_img(prediction)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Display the high redolution image in the first subplot
    axes[0].imshow(image1)
    axes[0].set_title('high redolution')

    # Display the low redolution image in the second subplot
    axes[1].imshow(image2)
    axes[1].set_title('low redolution')

    # Display the prediction image in the third subplot
    axes[2].imshow(image3)
    axes[2].set_title('prediction')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot 
    filename = os.path.basename(os.path.basename(prediction_img_path))
    plt.savefig(prediction_result_path+filename)
    
    print("PSNR of lowres images is %.4f" % (bicubic_psnr / 10))
    print("PSNR of reconstructions is %.4f" % (test_psnr / 10))

print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))