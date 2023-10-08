import os
from PIL import Image


def downsample_images(input_folder, factor=4):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)

                # Downsample the image
                img_resized = img.resize((img.width // factor, img.height // factor), Image.ANTIALIAS)
                img_resized.save(img_path)


if __name__ == "__main__":
    downsample_images("data/ad_nc/train")
    downsample_images("data/ad_nc/test")
