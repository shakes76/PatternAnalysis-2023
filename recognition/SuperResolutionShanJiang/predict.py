from utils import *
from modules import *
from dataset import *


from tensorflow import keras
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
import os
import math
import matplotlib.pyplot as plt

# load the trained model
checkpoint_filepath= "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/tmp/checkpoint/"
model = get_model()
model.load_weights(checkpoint_filepath)

#Get acees to path of each image
prediction_path = "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/prediction"
prediction_path = sorted(
    [
        os.path.join(prediction_path, fname)
        for fname in os.listdir(prediction_path)
        if fname.endswith(".jpeg")
    ]
)


# Dowansample resolution of iamges by factor of 4, then predict higher resolution image using the model
total_bicubic_psnr = 0.0 # PSNR of downsampled image
total_test_psnr = 0.0 # PSNR of model output

for index, prediction_img_path in enumerate(prediction_path[0:len(prediction_path)]):
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

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))