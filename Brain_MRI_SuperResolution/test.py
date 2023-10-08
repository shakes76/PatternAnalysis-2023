from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

model = load_model("saved_models/sub_pixel_cnn.h5")

# Load an image from the test set
img_path = 'data/ad_nc/test/AD/image1.png'
img = load_img(img_path, target_size=(128, 128))
img = img_to_array(img)/255.0
img = np.expand_dims(img, axis=0)

# Predict
super_res = model.predict(img)

# If needed, you can save or visualize the super_res output.
