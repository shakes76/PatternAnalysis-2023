import tensorflow as tf
import glob

def process_images(file_path, is_mask):
    # Decodes the image at the given file location
    if is_mask:
        image = tf.image.decode_png(tf.io.read_file(file_path), channels=1)
    else:
        image = tf.image.decode_jpeg(tf.io.read_file(file_path), channels=3)
    # Converts the image to float32
    image_converted = tf.image.convert_image_dtype(image, tf.float32)
    # Resizes the image to fit the given dimensions
    image_resized = tf.image.resize(image_converted, size=(img_height, img_width))
    # Normalises input image
    if is_mask:
        image_final = image_resized
    else:
        image_final = tf.cast(image_resized, tf.float32) / 255.0
    return image_final

def create_ds():
    # Calculates the size of each test, train, and validation subset
    files_ds_size = len(list(files_ds))
    train_ds_size = int(training_split * files_ds_size)
    val_ds_size = int(validation_split * files_ds_size)
    test_ds_size = files_ds_size - train_ds_size - val_ds_size
    # Prints the size of all the subsets
    print("Training size: %d" % train_ds_size)
    print("Validation size: %d" % val_ds_size)
    print("Testing size: %d" % test_ds_size)
    # Splits the dataset into test, validate, and train subsets
    train = files_ds.take(train_ds_size)
    val = files_ds.skip(train_ds_size).take(val_ds_size)
    test = files_ds.skip(train_ds_size).skip(val_ds_size)
    return train, val, test

img_height = img_width = 256

# Segments folders into arrays
image_file_list = list(glob.glob('ISIC2018_Task1-2_Training_Input/*.jpg'))
mask_file_list = list(glob.glob('ISIC2018_Task1_Training_GroundTruth/*.png'))

files_ds = tf.data.Dataset.from_tensor_slices((image_file_list, mask_file_list))
files_ds = files_ds.map(lambda x, y: (process_images(x, False), process_images(y, True)),
                        num_parallel_calls=tf.data.AUTOTUNE)

training_split = 0.8
validation_split = 0.1