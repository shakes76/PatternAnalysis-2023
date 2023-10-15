import os
import tensorflow as tf
import numpy as np
import random

AD_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//train//AD'
NC_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//train//NC'

AD_TEST_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//test//AD'
NC_TEST_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//test//NC'

def load_siamese_data(batch_size=32, train_ratio=0.8):
    # def preprocess_image(path):
    #     image = tf.io.read_file(path)
    #     image = tf.image.decode_jpeg(image, 1)
    #     image = tf.image.resize(image, [128, 128])  # You can adjust the size here
    #     return image / 255

    custom_ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    custom_nc_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]

    num_pairs = len(custom_ad_paths) // 3  # Modify the data splitting strategy

    custom_pair_base = custom_nc_paths[::3]
    custom_pair_ad = custom_ad_paths[:num_pairs]
    custom_pair_nc = custom_nc_paths[1::3][:num_pairs]

    random.shuffle(custom_pair_ad)
    random.shuffle(custom_pair_nc)

    custom_pair_compare = custom_pair_nc + custom_pair_ad

    custom_labels = np.concatenate([np.zeros([num_pairs]), np.ones([num_pairs])])
    custom_labels = np.expand_dims(custom_labels, -1)

    custom_base_ds = tf.data.Dataset.from_tensor_slices(custom_pair_base).map(preprocess_image)
    custom_pair_ds = tf.data.Dataset.from_tensor_slices(custom_pair_compare).map(preprocess_image)
    custom_labels_ds = tf.data.Dataset.from_tensor_slices(custom_labels)

    custom_dataset = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds), custom_labels_ds)).shuffle(num_pairs)

    train_num = int(round(len(custom_dataset) * train_ratio, 0))
    custom_train = custom_dataset.take(train_num)
    custom_val = custom_dataset.skip(train_num)

    return custom_train.batch(batch_size), custom_val.batch(batch_size)

# def load_classify_data(testing: bool, batch_size=32):
#     if not testing:
#         ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
#         cn_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]
#     else:
#         ad_paths = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
#         cn_paths = [os.path.join(NC_TEST_PATH, path) for path in os.listdir(NC_TEST_PATH)]

#     paths = ad_paths + cn_paths

#     labels = np.concatenate([np.ones([len(ad_paths)]), np.zeros([len(cn_paths)])])
#     labels = np.expand_dims(labels, -1)

#     images_ds = []

#     for path in paths:
#         image = tf.io.read_file(path)
#         image = tf.image.decode_jpeg(image, 1)
#         image = tf.image.resize(image, [128, 128])  # You can adjust the size here
#         image = image / 255
#         images_ds.append(image)

#     labels_ds = tf.data.Dataset.from_tensor_slices(labels)
#     images_ds = tf.data.Dataset.from_tensor_slices(images_ds)

#     dataset = tf.data.Dataset.zip((images_ds, labels_ds)).shuffle(len(paths))

#     if testing:
#         return dataset.batch(32)
#     else:
#         train_num = int(round(len(dataset) * 0.7, 0))
#         train = dataset.take(train_num)
#         val = dataset.skip(train_num)
#         return train.batch(batch_size), val.batch(batch_size)


def load_classify_data(testing: bool, batch_size=32): 
    if not testing:
        ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
        cn_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]
    else:
        ad_paths = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
        cn_paths = [os.path.join(NC_TEST_PATH, path) for path in os.listdir(NC_TEST_PATH)]

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
        return train.batch(batch_size), val.batch(batch_size)
    
def preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, 1)
        image = tf.image.resize(image, [128, 128])  # You can adjust the size here
        return image / 255