import tensorflow as tf
import glob

class SegmentationDataset(tf.data.Dataset):
    def _generator(image_paths, mask_paths, image_size):
        for image_path, mask_path in zip(image_paths, mask_paths):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = image / 255.0  # Normalize

            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)  # Assuming mask is in PNG format
            mask = tf.image.resize(mask, image_size)
            mask = mask / 255.0  # Normalize

            yield image, mask

    def __new__(cls, image_dir, mask_dir, image_size, cache=False):
        image_paths = sorted(glob.glob(image_dir + "/*"))
        mask_paths = sorted(glob.glob(mask_dir + "/*"))

        # Assume that the number of images and masks are the same
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(image_size[0], image_size[1], 1), dtype=tf.float32)
            ),
            args=(image_paths, mask_paths, image_size)
        )

        if cache:
            dataset = dataset.cache()

        return dataset

def get_dataloaders(batch_size=32, image_size=(572, 572)):
    train_dataset = SegmentationDataset("/datasets/ISIC2018_Task1-2_Training_Input", "/datasets/ISIC2018_Task1-2_Training_Input_GroundTruth", image_size, cache=True)
    test_dataset = SegmentationDataset("/datasets/ISIC2018_Task1-2_Test_Input", "/datasets/ISIC2018_Task1-2_Test_Input_GroundTruth", image_size, cache=True)
    valid_dataset = SegmentationDataset("/datasets/ISIC2018_Task1-2_Validation_Input", "/datasets/ISIC2018_Task1-2_Validation_Input_GroundTruth", image_size, cache=True)

    train_dataloader = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataloader = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_dataloader = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataloader, test_dataloader, valid_dataloader
