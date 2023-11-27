def loadData():
    """
    Loading the dataset.
    """
    trainData = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_LOAD_DEST + "/train", labels='inferred', label_mode='binary',
        image_size=[IMG_SIZE, IMG_SIZE], shuffle=True,
        batch_size=BATCH_SIZE, seed=8, class_names=['AD', 'NC']
    )

    testData = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_LOAD_DEST + "/test", labels='inferred', label_mode='binary',
        image_size=[IMG_SIZE, IMG_SIZE], shuffle=True,
        batch_size=BATCH_SIZE, seed=8, class_names=['AD', 'NC']
    )

    # Augmenting data
    normalize = tf.keras.layers.Normalization()
    flip = tf.keras.layers.RandomFlip(mode='horizontal', seed=8)
    rotate = tf.keras.layers.RandomRotation(factor=0.02, seed=8)
    zoom = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=8)

    trainData = trainData.map(
        lambda x, y: (rotate(flip(zoom(normalize(x)))), y)
    )

    testData = testData.map(
        lambda x, y: (rotate(flip(zoom(normalize(x)))), y)
    )

    # Taking half of the 9000 images from the test set as validation data
    validationData = testData.take(len(list(testData))//2)

    # Using remaining images as test set
    testData = testData.skip(len(list(testData))//2)

    return trainData, validationData, testData

