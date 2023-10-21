import tensorflow as tf
import glob
import cv2
import numpy as np

class SegmentationDataset(tf.data.Dataset):
    def __init__(self, imageDir, maskDir, image_size, cache=False):
        # store the image and mask filepaths, and augmentation transforms
        self.cache = cache
        self.imagePaths = sorted(glob.glob(imageDir + "/*"))
        self.maskPaths = sorted(glob.glob(maskDir + "/*"))
        self.image_size = image_size

        # Create a dataset from the list of file paths
        self.dataset = tf.data.Dataset.from_tensor_slices((self.imagePaths, self.maskPaths))
        self.dataset = self.dataset.map(self._load_image_and_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.cache:
            self.dataset = self.dataset.cache()

    def _load_image_and_mask(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = image / 255.0  # Normalize

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_jpeg(mask, channels=1)
        mask = tf.image.resize(mask, self.image_size)
        mask = mask / 255.0  # Normalize

        return image, mask

    def __call__(self, batch_size=32, shuffle=False):
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return self.dataset

def get_dataloaders(batch_size=32):
    train_dataset = SegmentationDataset("/datasets/ISIC2018_Task1-2_Training_Input",
                                        "/datasets/ISIC2018_Task1-2_Training_Input_GroundTruth",
                                        (572, 572), cache=True) 

    test_dataset = SegmentationDataset("/datasets/ISIC2018_Task1-2_Test_Input",
                                       "/datasets/ISIC2018_Task1-2_Test_Input_GroundTruth",
                                       (572, 572), cache=True)

    valid_dataset = SegmentationDataset("/datasets/ISIC2018_Task1-2_Training_Input",
                                        "/datasets/ISIC2018_Task1-2_Training_Input_GroundTruth",
                                        (572, 572), cache=True)

    return train_dataset(batch_size), test_dataset(batch_size), valid_dataset(batch_size)
