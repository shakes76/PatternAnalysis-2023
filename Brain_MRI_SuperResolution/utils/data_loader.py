from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import tensorflow as tf;
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

    # Print the directory we're attempting to load from
    print("Loading images from directory:", category_dir)

    for root, _, files in os.walk(category_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)

                # Print the path of the image we're attempting to load
                print("Loading image:", image_path)

                image = load_img(image_path, target_size=target_size, color_mode='grayscale')
                image_arr = img_to_array(image) / 255.0
                images.append(image_arr)
            else:
                # Print files that were skipped due to file type mismatch
                print("Skipped file (not an image or unsupported format):", file)

    # Print the total number of images loaded
    print(f"Total {category} images loaded:", len(images))

    return np.array(images)

# def load_data():
#     crop_size = 300
#     upscale_factor = 3
#     input_size = crop_size // upscale_factor
#     batch_size = 8
#
#     # --------- Load Datasets ---------
#     print(f"Attempting to load from: {train_dir}")  # Add this for diagnostics
#
#     train_ds = image_dataset_from_directory(
#         train_dir,
#         batch_size=batch_size,
#         image_size=(crop_size, crop_size),
#         validation_split=0.2,
#         subset="training",
#         seed=1337,
#         label_mode=None
#     )
#
#     print(f"Found {len(train_ds)} batches of training data.")
#
#     valid_ds = image_dataset_from_directory(
#         test_dir, batch_size=batch_size, image_size=(crop_size, crop_size),
#         validation_split=0.2, subset="validation", seed=1337, label_mode=None)
#
#     # --------- Scaling Function ---------
#     def scaling(input_image):
#         return input_image / 255.0
#
#     # Apply scaling to datasets
#     train_ds = train_ds.map(scaling)
#     valid_ds = valid_ds.map(scaling)
#
#     # --------- Input and Target Processing Functions ---------
#     def process_input(input, input_size, upscale_factor):
#         input = tf.image.rgb_to_yuv(input)
#         y, u, v = tf.split(input, 3, axis=-1)
#         return tf.image.resize(y, [input_size, input_size], method="area")
#
#     def process_target(input):
#         input = tf.image.rgb_to_yuv(input)
#         y, u, v = tf.split(input, 3, axis=-1)
#         return y
#
#     # Preprocess datasets by applying the above functions
#     def preprocess_datasets(train_ds, valid_ds, input_size, upscale_factor):
#         train_ds = train_ds.map(lambda x: (process_input(x, input_size, upscale_factor), process_target(x)))
#         train_ds = train_ds.prefetch(buffer_size=32)
#
#         valid_ds = valid_ds.map(lambda x: (process_input(x, input_size, upscale_factor), process_target(x)))
#         valid_ds = valid_ds.prefetch(buffer_size=32)
#
#         return train_ds, valid_ds
#
#     # Apply preprocessing to datasets
#     train_ds, valid_ds = preprocess_datasets(train_ds, valid_ds, input_size, upscale_factor)
#
#     return train_ds, valid_ds
#
#
# if __name__ == '__main__':
#     train_ds, valid_ds = load_data()
#
#
#     # --------- Visualization Function ---------
#     def visualize_samples(train_ds):
#         print("Visualizing a few sample training images:")
#         for batch in train_ds.take(1):
#             for img in batch[0]:
#                 array_to_img(img).show()
#             for img in batch[1]:
#                 array_to_img(img).show()
#
#
#     visualize_samples(train_ds)
#
#     # --------- Prepare Test Image Paths ---------
#     test_img_paths = sorted(
#         [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith(".jpeg")])
#     print(f"Found {len(test_img_paths)} test images.")
