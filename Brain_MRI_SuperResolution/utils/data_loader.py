from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np
from PIL import Image, ImageOps
from PIL import ImageEnhance

# Constants for file extensions and padding color
FILE_EXTENSIONS = ['.jpeg']
PADDING_COLOR = "white"  # Easily change padding color if needed


def sharpen_image(img, factor=2.0):
    """Enhance the sharpness of the image."""
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
    """Downsample the image by the given factor."""
    return img.resize((img.width // factor, img.height // factor), Image.LANCZOS)


def process_images(input_folder, output_folder, factor=4, display_sample=True):
    """Process and downsample the images in the given input folder and save them to the output folder."""

    # Create output directory if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = 0

    # Iterate through each image in the folder and process it
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

                    # Display the first processed image as a sample
                    if processed_files == 1 and display_sample:
                        img_resized.show()
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")




if __name__ == "__main__":
    process_images("AD_NC/train", "AD_NC/train_down sampled")
    process_images("AD_NC/test", "AD_NC/test_down sampled")


def load_images_from_category(base_dir, data_type, category, target_size=(100, 100)):
    """
    Load images from a specific category (AD or NC) within a specific data type (train or test).

    :param base_dir: Base directory containing the dataset folders.
    :param data_type: Type of data to load, either "train" or "test".
    :param category: Category of images to load, either "AD" or "NC".
    :param target_size: Tuple indicating the size to which images should be resized.
    :return: Numpy array containing the loaded images.
    """
    images = []
    category_dir = os.path.join(base_dir, data_type, category)

    for root, _, files in os.walk(category_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image = load_img(image_path, target_size=target_size, color_mode='grayscale')
                image_arr = img_to_array(image) / 255.0
                images.append(image_arr)
            else:
                # Print files that were skipped due to file type mismatch
                print("Skipped file (not an image or unsupported format):", file)

    return np.array(images)
