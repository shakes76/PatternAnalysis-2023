import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image

#TODO: Clean up debugging notes/code.
#      Make sure references are in order. Note use of tensorflow documentation.
#      Write README. Note where all files are saved in README.
#      Make pull request.

#NOTE: Expecting 2595 images in the training folders (-1 for license file) (should also be -1 for attribution file).
#      Getting 2596 though, not sure why.

#TODO: Change comments/spec to reference colour channels.
#Taken from code I wrote for prac2.
def loadDataFrom(directory, channels, size=128):
    """Loads the images from the given directory.

    Args:
      directory: The directory where the images are stored.

    Returns:
      NumPy array of shape (n, size, size), where n is the number
      of images in the directory.
    """
    numberOfImages = len(os.listdir(directory))
    if channels > 1:
        data = np.zeros((numberOfImages, size, size, channels))
    else:
        data = np.zeros((numberOfImages, size, size))
    data_shape = (numberOfImages, size, size, channels)
    #data = np.zeros(data_shape)
    for i, imageName in enumerate(sorted(os.listdir(directory))):
        #Make sure not to load in the license file.
        if imageName != "LICENSE.txt" and imageName != "ATTRIBUTION.txt":
            imagePath = os.path.join(directory, imageName)
            #image = Image.open(imagePath)
            #NOTE: Mention in the readme that doing this leaves some images as only skin cancer, and some as none.
            #      If cropping isn't used mention squashed aspect ratios.
            #Center crop then resize to maintain aspect ratio.
            #Center crop algorithm from:
            #https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
            #crop_size = 256
            #width, height = image.size
            #left = (width - crop_size) // 2
            #top = (height - crop_size) // 2
            #right = left + crop_size
            #bottom = top + crop_size
            #image = image.crop((left, top, right, bottom)).resize((size, size))
            #NOTE: Unsure if resizing should be done in this function.
            if channels > 1:
                data[i, :, :, :] = np.asarray(Image.open(imagePath).resize((size, size)))
                #data[i, :, :, :] = np.asarray(image)
            else:
                data[i, :, :] = np.asarray(Image.open(imagePath).resize((size, size)))
                #data[i, :, :] = np.asarray(image)
                #data[i, :, :, :] = np.reshape(np.asarray(Image.open(imagePath).resize((size, size))), data_shape)
    #Need the extra dimension for image manipulation.
    if channels == 1:
        data = np.reshape(data, data_shape)
    return tf.data.Dataset.from_tensor_slices(data)

#The following is modified code from:
#https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def partition(data, train_size, val_size, test_size, seed):
    #TODO: Write specification.
    data.shuffle(2596, seed)
    print("AFTER SHUFFLE")
    print(data.element_spec)
    train_data = data.take(train_size)
    val_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size).skip(val_size)
    return train_data, val_data, test_data

#NOTE: Is some other form of normalization needed for masks?
#      e.g. Set all 255s to 1s, and everything else to 0s.
#      Could use tf.clip_by_value for this.

#Taken from code I wrote for prac2.
def normalize(image, mask):
    #TODO: Write specification.
    #NOTE: Trying not casting, should remove extra dimension.
    image = tf.cast(image, tf.float64) / 255.0
    print(type(image))
    #NOTE: Could reintroduce cast now that tuple is dealt with.
    #image = image / 255.0
    #NOTE: Should some kind of softmax be used here instead?
    #mask = tf.cast(mask, tf.float64) / 255.0
    #mask = mask / 255.0
    mask = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=1)
    #NOTE: Squeeze in here to get rid of useless dim?
    #image = tf.squeeze(image)
    #mask = tf.squeeze(mask)
    #Convert mask to one hot encoding.
    #mask = tf.cast(mask, tf.int8)
    #mask = tf.one_hot(mask, 2)
    #mask = tf.squeeze(mask)
    return image, mask

#Based on code from:
#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
def augment(image, mask):
    #TODO: Write specification.
    p = tf.random.uniform(())
    if p < 0.25:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    elif p < 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    #NOTE: Squeeze in here to get rid of useless dim?
    #image = tf.squeeze(image)
    #mask = tf.squeeze(mask)
    return image, mask

#NOTE: Trying to get rid of these extra dimensions.
#      Didn't seem to work, get rid of it.
def tuple_squeeze(image, mask):
    #TODO: Write specification.
    image = tf.squeeze(image)
    mask = tf.squeeze(mask)
    return image, mask

#NOTE: Unsure what the batch size should be.
def preprocessing(batch_size=64):
    #TODO: Write specification.
    #These are the directories for the datasets.
    test_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input/"
    training_images_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/"
    training_gt_dir = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/"
    #Load in data.
    #NOTE: test_dir is unused, there don't seem to be segmentation maps in there.
    #NOTE: Trying tensorflow dataset loader.
    image_data = loadDataFrom(training_images_dir, channels=3)
    mask_data = loadDataFrom(training_gt_dir, channels=1)
    #image_data = tf.keras.utils.image_dataset_from_directory(
    #    training_images_dir,
    #    image_size=(128, 128))
    #mask_data = tf.keras.utils.image_dataset_from_directory(
    #    training_gt_dir,
    #    image_size=(128, 128))
    #NOTE: Getting weird shapes at training time.
    #      Tried squeezing data, but that didn't work.
    #image_data = image_data.map(tf.squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #mask_data = mask_data.map(tf.squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    isic_data = tf.data.Dataset.zip((image_data, mask_data))
    isic_data = isic_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("AFTER ZIP")
    print(isic_data.element_spec)
    #Split.
    train_data, val_data, test_data = partition(isic_data, 1796, 400, 400, seed=271828)
    print("AFTER PARTITION")
    print(train_data.element_spec)
    print(val_data.element_spec)
    print(test_data.element_spec)
    #NOTE: How to ensure augment happens on a batch by batch basis, but normalize doesn't?
    #      Probably want to add layers to front of model that do data augmentation, rather than doing it in
    #      preprocessing. See "tensorflow data augmentation". Would have to be layers that are only used in training.
    #train_data = train_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #NOTE: Augmentation is unnecessary here if augment layer is used in model.
    #train_data = train_data.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #val_data = val_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #test_data = test_data.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #Create batches for training data.
    train_batches = train_data.cache().batch(batch_size).repeat()
    train_batches = train_batches.prefetch(tf.data.experimental.AUTOTUNE)
    val_batches = val_data.batch(batch_size)
    test_batches = test_data.batch(batch_size)
    #NOTE: Mapping is adding a dimension, but why?
    #      Also looks like squeezing bypasses tensorflow dataset shape info, so probs shouldn't be used here.
    #train_batches = train_batches.map(tuple_squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #val_data = val_data.map(tuple_squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #test_data = test_data.map(tuple_squeeze, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("AFTER MAPS")
    print(train_batches.element_spec)
    print(val_batches.element_spec)
    print(test_batches.element_spec)
    #NOTE: Is any other preprocessing needed?
    return train_batches, val_batches, test_batches

