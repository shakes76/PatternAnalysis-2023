import tensorflow as tf

class SkinLesionDataset:
    def __init__(self, data_dir, target_size=(512, 512)):
        self.data_dir = data_dir
        self.target_size = target_size

        self.train_dataset = self.load_dataset("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth")
        self.test_dataset = self.load_dataset("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth")
        self.validation_dataset = self.load_dataset("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth")

    def load_and_preprocess(self, image_path, mask_path):
        # Load and decode image
        image = tf.io.decode_image(tf.io.read_file(image_path), channels=3)  # Adjust channels as needed

        # Resize image to the target size with padding
        image = tf.image.resize_with_pad(image, target_height=self.target_size[0], target_width=self.target_size[1])
        image = tf.cast(image, tf.float32) / 255.0

        # Load and decode mask
        mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1)  # Assuming grayscale masks

        # Resize mask to the target size with padding
        mask = tf.image.resize_with_pad(mask, target_height=self.target_size[0], target_width=self.target_size[1])
        mask = tf.cast(mask, tf.float32) / 255.0

        return image, mask

    def load_dataset(self, input_folder, ground_truth_folder):
        input_paths = sorted([str(path.numpy()) for path in tf.data.Dataset.list_files(f"{self.data_dir}/{input_folder}/*")])
        ground_truth_paths = sorted([str(path.numpy()) for path in tf.data.Dataset.list_files(f"{self.data_dir}/{ground_truth_folder}/*")])

        dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(input_paths), tf.data.Dataset.from_tensor_slices(ground_truth_paths))).map(self.load_and_preprocess)

        return dataset

# Example usage:
# dataset = SkinLesionDataset("datasets")
# train_data = dataset.train_dataset
# test_data = dataset.test_dataset
# validation_data = dataset.validation_dataset
