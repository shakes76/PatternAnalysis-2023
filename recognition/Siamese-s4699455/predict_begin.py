import numpy as np
from PIL import Image

from predict import Siamese

image_1_path = "datasets/AD_NC/test/AD/400436_75.jpeg"
image_2_path = "datasets/AD_NC/test/AD/400436_76.jpeg"
image_3_path = "datasets/AD_NC/test/NC/1302056_88.jpeg"
if __name__ == "__main__":
    model = Siamese()
    image_1 = Image.open(image_1_path)
    image_2 = Image.open(image_2_path)
    image_3 = Image.open(image_3_path)

    probability12 = model.detect_image(image_1,image_2)
    probability13 = model.detect_image(image_1,image_3)

    print(image_1_path + " and " + image_2_path +" probability: " + str(probability12.tolist()))
    print(image_1_path + " and " + image_2_path +" probability: " + str(probability13.tolist()))
'''      
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.detect_image(image_1,image_2)
        print(probability)
'''  