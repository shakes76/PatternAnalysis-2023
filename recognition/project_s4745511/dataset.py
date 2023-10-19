import os
import tensorflow as tf
import numpy as np
import random

AD_TRAIN_DATA = "AD_NC/train/AD"
NC_TRAIN_DATA = "AD_NC/train/NC"

AD_TEST_DATA = "AD_NC/test/AD"
NC_TEST_DATA = "AD_NC/test/NC"

def siamese_data_loader(batch_size=32, train_ratio=0.8):
    def preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, 1)
        image = tf.image.resize(image, [128, 128])  # You can adjust the size here
        return image / 255
    custom_ad_paths = [os.path.join(AD_TRAIN_DATA, path) for path in os.listdir(AD_TRAIN_DATA)]
    custom_nc_paths = [os.path.join(NC_TRAIN_DATA, path) for path in os.listdir(NC_TRAIN_DATA)]

    # Shuffle the paths
    random.shuffle(custom_ad_paths)
    random.shuffle(custom_nc_paths)

    num_pairs = min(len(custom_ad_paths), len(custom_nc_paths)) // 3  # Modify the data splitting strategy


    custom_pair_base_ad_ad = custom_ad_paths[::3][:num_pairs]
    custom_pair_ad_ad = custom_ad_paths[1::3][:num_pairs]
    custom_labels_ad_ad = np.ones([num_pairs])

    custom_pair_base_ad_nc = custom_ad_paths[::3][:num_pairs]
    custom_pair_ad_nc = custom_ad_paths[1::3][:num_pairs]
    custom_pair_nc_ad = custom_nc_paths[1::3][:num_pairs]
    custom_pair_compare_ad_nc = custom_pair_ad_nc + custom_pair_nc_ad
    custom_labels_ad_nc = np.zeros([num_pairs])

    custom_pair_base_nc_nc = custom_nc_paths[::3][:num_pairs]
    custom_pair_nc_nc = custom_nc_paths[1::3][:num_pairs]
    custom_labels_nc_nc = np.ones([num_pairs])

    custom_base_ds = tf.data.Dataset.from_tensor_slices(custom_pair_base_ad_ad + custom_pair_base_ad_nc + custom_pair_base_nc_nc).map(preprocess_image)
    custom_pair_ds = tf.data.Dataset.from_tensor_slices(custom_pair_ad_ad + custom_pair_compare_ad_nc + custom_pair_nc_nc).map(preprocess_image)
    custom_labels_ds = tf.data.Dataset.from_tensor_slices(
        np.concatenate([custom_labels_ad_ad, custom_labels_ad_nc, custom_labels_nc_nc])
    )

    custom_dataset = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds), custom_labels_ds)).shuffle(len(custom_pair_base_ad_ad) + len(custom_pair_base_ad_nc) + len(custom_pair_base_nc_nc))

    train_num = int(round(len(custom_dataset) * train_ratio, 0))
    custom_train = custom_dataset.take(train_num)
    custom_val = custom_dataset.skip(train_num)

    return custom_train.batch(batch_size), custom_val.batch(batch_size)

def classification_data_loader(testing: bool):
    def preprocess_image(path):
          image = tf.io.read_file(path)
          image = tf.image.decode_jpeg(image, 1)
          image = tf.image.resize(image, [128, 128])  # You can adjust the size here
          return image / 255
    
    # batch_size=32

    if not testing:
        ad_paths = [os.path.join(AD_TRAIN_DATA, path) for path in os.listdir(AD_TRAIN_DATA)]
        cn_paths = [os.path.join(NC_TRAIN_DATA, path) for path in os.listdir(NC_TRAIN_DATA)]
    else:
        ad_paths = [os.path.join(AD_TEST_DATA, path) for path in os.listdir(AD_TEST_DATA)]
        cn_paths = [os.path.join(NC_TEST_DATA, path) for path in os.listdir(NC_TEST_DATA)]

    paths = ad_paths + cn_paths

    labels = np.concatenate([np.ones([len(ad_paths)]), np.zeros([len(cn_paths)])])
    labels = np.expand_dims(labels, -1)

    images_ds = tf.data.Dataset.from_tensor_slices(paths).map(preprocess_image)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images_ds, labels_ds)).shuffle(len(paths))

    if testing:
        return dataset.batch(32)
    else:
        train_num = int(round(len(dataset) * 0.7, 0))
        train = dataset.take(train_num)
        val = dataset.skip(train_num)
        return train.batch(32), val.batch(32)