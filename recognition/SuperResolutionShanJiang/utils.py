import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL
import numpy as np
from keras.utils import img_to_array

# downsamples given image by  ratio of given upscale_factor.
def get_lowres_image(img, upscale_factor):
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


# preprocessed given image and use the give model to increase its resolution
def upscale_image(model, img):
    ycbcr = img.convert("YCbCr") # Convert image to YCbCr colout spave
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0 # Normalise the pixel values 

    input = np.expand_dims(y, axis=0)
    out = model.predict(input) 

    out_img_y = out[0]
    out_img_y *= 255.0 # Denormalise the pixel values 

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


