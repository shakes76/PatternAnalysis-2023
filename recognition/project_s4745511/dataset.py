import os
import tensorflow as tf
import numpy as np
import random

AD_TRAIN_DATA = "AD_NC/train/AD"
NC_TRAIN_DATA = "AD_NC/train/NC"

AD_TEST_DATA = "AD_NC/test/AD"
NC_TEST_DATA = "AD_NC/test/NC"

def siamese_data_loader(batch_size=32, train_ratio=0.8):
    """
    Function: Data loader for siamese, 
    Input Parameters:
        - batch_size: An integer specifying the batch size for the returned datasets (default is 32).
        - train_ratio: A float specifying the ratio of data to be used for training (default is 0.8).
    Returns:
        - custom_train: A TensorFlow Dataset containing training data (image pairs and labels) with the specified batch size.
        - custom_val: A TensorFlow Dataset containing validation data (image pairs and labels) with the specified batch size.
    """
    def preprocess_image(path):
        """
        Function: Image Processing and Resizing (Nested function)
        Imput Parameter: Takes the path of the image as input
        Return Type: Resized image
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, 1)
        image = tf.image.resize(image, [128, 128])  # You can adjust the size here
        return image / 255
    
    custom_ad_paths = [os.path.join(AD_TRAIN_DATA, path) for path in os.listdir(AD_TRAIN_DATA)]
    custom_nc_paths = [os.path.join(NC_TRAIN_DATA, path) for path in os.listdir(NC_TRAIN_DATA)]

    # Radom Shuffle before generating the pairs
    random.shuffle(custom_ad_paths)
    random.shuffle(custom_nc_paths)

    num_pairs = min(len(custom_ad_paths), len(custom_nc_paths)) // 3  # Modify the data splitting strategy

    #creating pairs - ad-ad combination (1/positive pair)
    custom_pair_base_ad_ad = custom_ad_paths[::3][:num_pairs]
    custom_pair_ad_ad = custom_ad_paths[1::3][:num_pairs]
    custom_labels_ad_ad = np.ones([num_pairs])

    #creating pairs - nc-ad combination and ad-nc (0/negative pair)
    custom_pair_base_ad_nc = custom_ad_paths[::3][:num_pairs]
    custom_pair_ad_nc = custom_ad_paths[1::3][:num_pairs]
    custom_pair_nc_ad = custom_nc_paths[1::3][:num_pairs]
    custom_pair_compare_ad_nc = custom_pair_ad_nc + custom_pair_nc_ad
    custom_labels_ad_nc = np.zeros([num_pairs])

    #creating pairs - nc-nc combination (1/positive  pair)
    custom_pair_base_nc_nc = custom_nc_paths[::3][:num_pairs]
    custom_pair_nc_nc = custom_nc_paths[1::3][:num_pairs]
    custom_labels_nc_nc = np.ones([num_pairs])

    custom_base_ds = tf.data.Dataset.from_tensor_slices(custom_pair_base_ad_ad + custom_pair_base_ad_nc + custom_pair_base_nc_nc).map(preprocess_image)
    custom_pair_ds = tf.data.Dataset.from_tensor_slices(custom_pair_ad_ad + custom_pair_compare_ad_nc + custom_pair_nc_nc).map(preprocess_image)
    custom_labels_ds = tf.data.Dataset.from_tensor_slices(
        np.concatenate([custom_labels_ad_ad, custom_labels_ad_nc, custom_labels_nc_nc])
    )

    custom_dataset = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds), custom_labels_ds)).shuffle(len(custom_pair_base_ad_ad) + len(custom_pair_base_ad_nc) + len(custom_pair_base_nc_nc))

    #splitting the data for training amd validation
    train_num = int(round(len(custom_dataset) * train_ratio, 0))
    custom_train = custom_dataset.take(train_num)
    custom_val = custom_dataset.skip(train_num)

    return custom_train.batch(batch_size), custom_val.batch(batch_size)

def classification_data_loader(testing: bool):
    """
    Function: data loader for classification
    Input Parameters:
    - testing: A boolean indicating whether to load data for testing (True) or training (False).
    Returns:
    - dataset: A TensorFlow Dataset containing image data and labels (for testing) with a batch size of 32.
    - train: A TensorFlow Dataset containing training data (image data and labels) with a batch size of 32.
    - val: A TensorFlow Dataset containing validation data (image data and labels) with a batch size of 32.
    """
    def preprocess_image(path):
        """
        Function: Image Processing and Resizing
        Imput Parameter: Takes the path of the image as input
        Return Type: Resized image
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, 1)
        image = tf.image.resize(image, [128, 128])  # You can adjust the size here
        return image / 255
    
    if not testing:
    # Load training data paths for "AD" and "NC" classes.
        ad_paths = [os.path.join(AD_TRAIN_DATA, path) for path in os.listdir(AD_TRAIN_DATA)]
        cn_paths = [os.path.join(NC_TRAIN_DATA, path) for path in os.listdir(NC_TRAIN_DATA)]
    else:
        # Load testing data paths for "AD" and "NC" classes.
        ad_paths = [os.path.join(AD_TEST_DATA, path) for path in os.listdir(AD_TEST_DATA)]
        cn_paths = [os.path.join(NC_TEST_DATA, path) for path in os.listdir(NC_TEST_DATA)]

    # Combine paths for both classes.
    paths = ad_paths + cn_paths

    # Create labels: 1 for "AD" and 0 for "NC," then expand dimensions.
    labels = np.concatenate([np.ones([len(ad_paths)]), np.zeros([len(cn_paths)])])
    labels = np.expand_dims(labels, -1)

    # Create a TensorFlow Dataset of image paths and preprocess images.
    images_ds = tf.data.Dataset.from_tensor_slices(paths).map(preprocess_image)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    # Combine image and label datasets and shuffle the data.
    dataset = tf.data.Dataset.zip((images_ds, labels_ds)).shuffle(len(paths))

    if testing:
        # If testing, return the entire dataset batched into 32-sized batches.
        return dataset.batch(32)
    else:
        # If not testing, perform an 80-20 train-validation split.
        train_num = int(round(len(dataset) * 0.8, 0))
        train = dataset.take(train_num)
        val = dataset.skip(train_num)
        
        # Return the training and validation datasets, each batched into 32-sized batches.
        return train.batch(32), val.batch(32)
