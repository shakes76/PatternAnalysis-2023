import os
import tensorflow as tf
from tensorflow import keras

class DataLoader:
    """
    A utility class for creating data generators for image segmentation tasks.
    It prepares and generates batches of input images and corresponding ground truth masks
    for training or validation purposes.

    Parameters:
        - input_dir: The directory containing input images.
        - groundtruth_dir: The directory containing corresponding ground truth masks.
        - image_mode: The color mode for input images (e.g., 'rgb').
        - mask_mode: The color mode for ground truth masks (e.g., 'grayscale').
        - image_height: The desired height of the input images.
        - image_width: The desired width of the input images.
        - batch_size: The batch size for data generation.
        - seed: The random seed for data augmentation (default is 45).
        - shear_range: The range for shear transformations during data augmentation.
        - zoom_range: The range for zooming transformations during data augmentation.
        - horizontal_flip: Whether to perform horizontal flips during data augmentation.
        - vertical_flip: Whether to perform vertical flips during data augmentation.
        - fill_mode: The fill mode for filling in newly created pixels during data augmentation.

    Methods:
        - create_data_generators: Creates and returns data generators for input images and masks.
    """
    def __init__(self, input_dir, groundtruth_dir, image_mode, mask_mode, image_height, image_width, batch_size, seed=45, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode='nearest',):
        self.input_dir = input_dir
        self.groundtruth_dir = groundtruth_dir
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.seed = seed
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode

    def create_data_generators(self):
        data_gen_args = dict(
            rescale=1.0 / 255,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            fill_mode=self.fill_mode,)

        input_image_generator = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
        groundtruth_mask_generator = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

        input_gen = input_image_generator.flow_from_directory(
            self.input_dir,
            color_mode=self.image_mode,
            seed=self.seed,
            class_mode=None,
            batch_size=self.batch_size,
            interpolation="nearest",
            target_size=(self.image_height, self.image_width))

        groundtruth_gen = groundtruth_mask_generator.flow_from_directory(
            self.groundtruth_dir,
            color_mode=self.mask_mode,
            seed=self.seed,
            class_mode=None,
            batch_size=self.batch_size,
            interpolation="nearest",
            target_size=(self.image_height, self.image_width))

        return zip(input_gen, groundtruth_gen)
