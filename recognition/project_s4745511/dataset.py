import os
import tensorflow as tf
import numpy as np
import random

AD_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//train//AD'
NC_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//train//NC'

AD_TEST_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//test//AD'
NC_TEST_PATH = 'C://Users//Danita//OneDrive//Desktop//UQ//sem3//PatterRecognision//Project3//ADNI_AD_NC_2D//AD_NC//test//NC'

# def load_siamese_data(batch_size=32, train_ratio=0.8):
#     custom_ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
#     custom_nc_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]

#     num_pairs = min(len(custom_ad_paths), len(custom_nc_paths)) // 3  # Modify the data splitting strategy

    
#     custom_pair_base_ad_ad = custom_ad_paths[::3][:num_pairs]
#     custom_pair_ad_ad = custom_ad_paths[1::3][:num_pairs]
#     custom_labels_ad_ad = np.ones([num_pairs])

#     custom_pair_base_ad_nc = custom_ad_paths[::3][:num_pairs]
#     custom_pair_ad_nc = custom_ad_paths[1::3][:num_pairs]
#     custom_pair_nc_ad = custom_nc_paths[1::3][:num_pairs]
#     custom_pair_compare_ad_nc = custom_pair_ad_nc + custom_pair_nc_ad
#     custom_labels_ad_nc = np.zeros([num_pairs])

#     custom_pair_base_nc_nc = custom_nc_paths[::3][:num_pairs]
#     custom_pair_nc_nc = custom_nc_paths[1::3][:num_pairs]
#     custom_labels_nc_nc = np.ones([num_pairs])

#     custom_base_ds = tf.data.Dataset.from_tensor_slices(custom_pair_base_ad_ad + custom_pair_base_ad_nc + custom_pair_base_nc_nc).map(preprocess_image)
#     custom_pair_ds = tf.data.Dataset.from_tensor_slices(custom_pair_ad_ad + custom_pair_compare_ad_nc + custom_pair_nc_nc).map(preprocess_image)
#     custom_labels_ds = tf.data.Dataset.from_tensor_slices(
#         np.concatenate([custom_labels_ad_ad, custom_labels_ad_nc, custom_labels_nc_nc])
#     )

#     custom_dataset = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds), custom_labels_ds)).shuffle(len(custom_pair_base_ad_ad) + len(custom_pair_base_ad_nc) + len(custom_pair_base_nc_nc))

#     train_num = int(round(len(custom_dataset) * train_ratio, 0))
#     custom_train = custom_dataset.take(train_num)
#     custom_val = custom_dataset.skip(train_num)

#     return custom_train.batch(batch_size), custom_val.batch(batch_size)




# def load_siamese_data(batch_size=32, train_ratio=0.8):
#     custom_ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
#     custom_nc_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]

#     num_pairs = len(custom_ad_paths) // 3  # Modify the data splitting strategy

#     # Create pairs for "AD, NC" combination
#     custom_pair_base = custom_nc_paths[::3]
#     custom_pair_ad = custom_ad_paths[:num_pairs]
#     custom_pair_nc = custom_nc_paths[1::3][:num_pairs]

#     random.shuffle(custom_pair_ad)
#     random.shuffle(custom_pair_nc)

#     custom_pair_compare_ad_nc = custom_pair_ad + custom_pair_nc
#     custom_labels_ad_nc = np.concatenate([np.zeros([num_pairs]), np.ones([num_pairs])])
#     custom_labels_ad_nc = np.expand_dims(custom_labels_ad_nc, -1)

#     # Create pairs for "NC, AD" combination
#     custom_pair_base_nc_ad = custom_ad_paths[::3]
#     custom_pair_ad_nc = custom_ad_paths[1::3][:num_pairs]
#     custom_pair_nc_ad = custom_nc_paths[:num_pairs]

#     random.shuffle(custom_pair_ad_nc)
#     random.shuffle(custom_pair_nc_ad)

#     custom_pair_compare_nc_ad = custom_pair_nc_ad + custom_pair_ad_nc
#     custom_labels_nc_ad = np.concatenate([np.zeros([num_pairs]), np.ones([num_pairs])])
#     custom_labels_nc_ad = np.expand_dims(custom_labels_nc_ad, -1)

#     # Combine both datasets to include all four combinations
#     custom_base_ds = tf.data.Dataset.from_tensor_slices(custom_pair_base + custom_pair_base_nc_ad).map(preprocess_image)
#     custom_pair_ds = tf.data.Dataset.from_tensor_slices(custom_pair_compare_ad_nc + custom_pair_compare_nc_ad).map(preprocess_image)
#     custom_labels_ds = tf.data.Dataset.from_tensor_slices(
#         np.concatenate([custom_labels_ad_nc, custom_labels_nc_ad])
#     )

#     custom_dataset = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds), custom_labels_ds)).shuffle(len(custom_pair_base) + len(custom_pair_base_nc_ad))

#     train_num = int(round(len(custom_dataset) * train_ratio, 0))
#     custom_train = custom_dataset.take(train_num)
#     custom_val = custom_dataset.skip(train_num)

#     return custom_train.batch(batch_size), custom_val.batch(batch_size)



# def load_siamese_data(batch_size=32, train_ratio=0.8): working code
#     custom_ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
#     custom_nc_paths = [os.path.join(NC_PATH, path) for path in os.listdir(NC_PATH)]

#     num_pairs = len(custom_ad_paths) // 3  # Modify the data splitting strategy

#     custom_pair_base = custom_nc_paths[::3]
#     custom_pair_ad = custom_ad_paths[:num_pairs]
#     custom_pair_nc = custom_nc_paths[1::3][:num_pairs]

#     random.shuffle(custom_pair_ad)
#     random.shuffle(custom_pair_nc)

#     custom_pair_compare_ad_nc = custom_pair_ad + custom_pair_nc
#     custom_labels_ad_nc = np.concatenate([np.zeros([num_pairs]), np.ones([num_pairs])])
#     custom_labels_ad_nc = np.expand_dims(custom_labels_ad_nc, -1)

#     # For the other combination (NC, AD), swap the order of custom_pair_ad and custom_pair_nc
#     custom_pair_compare_nc_ad = custom_pair_nc + custom_pair_ad
#     custom_labels_nc_ad = np.concatenate([np.zeros([num_pairs]), np.ones([num_pairs])])
#     custom_labels_nc_ad = np.expand_dims(custom_labels_nc_ad, -1)

#     custom_base_ds = tf.data.Dataset.from_tensor_slices(custom_pair_base).map(preprocess_image)
#     custom_pair_ds_ad_nc = tf.data.Dataset.from_tensor_slices(custom_pair_compare_ad_nc).map(preprocess_image)
#     custom_labels_ds_ad_nc = tf.data.Dataset.from_tensor_slices(custom_labels_ad_nc)
#     custom_dataset_ad_nc = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds_ad_nc), custom_labels_ds_ad_nc)).shuffle(num_pairs)

#     # Create another dataset for the other combination (NC, AD)
#     custom_pair_ds_nc_ad = tf.data.Dataset.from_tensor_slices(custom_pair_compare_nc_ad).map(preprocess_image)
#     custom_labels_ds_nc_ad = tf.data.Dataset.from_tensor_slices(custom_labels_nc_ad)
#     custom_dataset_nc_ad = tf.data.Dataset.zip(((custom_base_ds, custom_pair_ds_nc_ad), custom_labels_ds_nc_ad)).shuffle(num_pairs)

#     # Combine both datasets to include all four combinations
#     custom_dataset = custom_dataset_ad_nc.concatenate(custom_dataset_nc_ad)

#     train_num = int(round(len(custom_dataset) * train_ratio, 0))
#     custom_train = custom_dataset.take(train_num)
#     custom_val = custom_dataset.skip(train_num)

#     return custom_train.batch(batch_size), custom_val.batch(batch_size)


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