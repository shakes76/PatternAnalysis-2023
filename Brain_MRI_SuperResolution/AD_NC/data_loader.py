import os
import tensorflow as tf
from keras.preprocessing.image import array_to_img
from keras.preprocessing import image_dataset_from_directory

data_loader_directory = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(data_loader_directory, 'train_downsampled')
test_dir = os.path.join(data_loader_directory, 'test_downsampled')
def load_data():
    # --------- Specify Dataset Directories ---------


    # --------- Image Processing Parameters ---------
    crop_size = 300
    upscale_factor = 3
    input_size = crop_size // upscale_factor
    batch_size = 8

    # --------- Load Datasets ---------
    print(f"Attempting to load from: {train_dir}")  # Add this for diagnostics

    train_ds = image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None
    )

    print(f"Found {len(train_ds)} batches of training data.")

    valid_ds = image_dataset_from_directory(
        test_dir, batch_size=batch_size, image_size=(crop_size, crop_size),
        validation_split=0.2, subset="validation", seed=1337, label_mode=None)

    # --------- Scaling Function ---------
    def scaling(input_image):
        return input_image / 255.0

    # Apply scaling to datasets
    train_ds = train_ds.map(scaling)
    valid_ds = valid_ds.map(scaling)

    # --------- Input and Target Processing Functions ---------
    def process_input(input, input_size, upscale_factor):
        input = tf.image.rgb_to_yuv(input)
        y, u, v = tf.split(input, 3, axis=-1)
        return tf.image.resize(y, [input_size, input_size], method="area")

    def process_target(input):
        input = tf.image.rgb_to_yuv(input)
        y, u, v = tf.split(input, 3, axis=-1)
        return y

    # Preprocess datasets by applying the above functions
    def preprocess_datasets(train_ds, valid_ds, input_size, upscale_factor):
        train_ds = train_ds.map(lambda x: (process_input(x, input_size, upscale_factor), process_target(x)))
        train_ds = train_ds.prefetch(buffer_size=32)

        valid_ds = valid_ds.map(lambda x: (process_input(x, input_size, upscale_factor), process_target(x)))
        valid_ds = valid_ds.prefetch(buffer_size=32)

        return train_ds, valid_ds

    # Apply preprocessing to datasets
    train_ds, valid_ds = preprocess_datasets(train_ds, valid_ds, input_size, upscale_factor)

    return train_ds, valid_ds


if __name__ == '__main__':
    train_ds, valid_ds = load_data()


    # --------- Visualization Function ---------
    def visualize_samples(train_ds):
        print("Visualizing a few sample training images:")
        for batch in train_ds.take(1):
            for img in batch[0]:
                array_to_img(img).show()
            for img in batch[1]:
                array_to_img(img).show()


    visualize_samples(train_ds)

    # --------- Prepare Test Image Paths ---------
    test_img_paths = sorted(
        [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith(".jpeg")])
    print(f"Found {len(test_img_paths)} test images.")
