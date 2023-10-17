import tensorflow as tf

batch_size = 128
img_height = 240
img_width = 256

train = tf.keras.utils.image_dataset_from_directory(
  "D:/COMP3710 Project/Siamese/recognition/AD_NC/train",
  labels="inferred",
  image_size=(img_height, img_width),
  batch_size=batch_size)

test = tf.keras.utils.image_dataset_from_directory(
  "D:/COMP3710 Project/Siamese/recognition/AD_NC/test",
  labels="inferred",
  image_size=(img_height, img_width),
  batch_size=batch_size)

def get_data():
    return (train, test)