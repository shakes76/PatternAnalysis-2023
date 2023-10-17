import os

import numpy as np
from PIL import Image, ImageOps
from PIL import ImageEnhance

FILE_EXTENSIONS = ['.jpeg']
PADDING_COLOR = "white"  # Easily change padding color if needed


def sharpen_image(img, factor=2.0):
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def pad_image(img, desired_size):
    # Calculate the difference between the desired size and the image's size
    delta_width = desired_size[0] - img.width
    delta_height = desired_size[1] - img.height
    padding = (
        delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    return ImageOps.expand(img, padding, fill="white")


def downsample_image(img, factor=4):
    return img.resize((img.width // factor, img.height // factor), Image.LANCZOS)


def process_images(input_folder, output_folder, factor=4, display_sample=True):
    print(f"Processing images in folder: {input_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if any(file.endswith(ext) for ext in FILE_EXTENSIONS):
                img_path = os.path.join(root, file)
                try:
                    print(f"Processing image: {img_path}")

                    img = Image.open(img_path)
                    img_resized = downsample_image(img, factor)
                    img_resized = sharpen_image(img_resized)
                    save_path = os.path.join(output_folder, file)
                    img_resized.save(save_path)

                    processed_files += 1
                    print(f"Processed {processed_files}")

                    if processed_files == 1 and display_sample:
                        img_resized.show()
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print(f"Completed processing images in folder: {input_folder}")

if __name__ == "__main__":
    process_images("train", "train_down sampled")
    process_images("test", "test_down sampled")

