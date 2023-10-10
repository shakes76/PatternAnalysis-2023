import tensorflow as tf


def load_dataset(path: str) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the dataset from the given path
    :param path: Path to the dataset
    :return: train_ds, val_ds, test_ds
    """
    print(f"Loading AD training data")
    train_ad_ds = tf.keras.utils.image_dataset_from_directory(
        path + "/train/AD",
        labels=None,
        label_mode=None,
        image_size=(60, 64),
        batch_size=None,
        seed=321,
        color_mode="grayscale",
        shuffle=True,
    )

    train_ad_ds = train_ad_ds.map(lambda x: x / 255.0)

    print(f"Loading NC training data")
    train_nc_ds = tf.keras.utils.image_dataset_from_directory(
        path + "/train/NC",
        labels=None,
        label_mode=None,
        image_size=(60, 64),
        batch_size=None,
        seed=321,
        color_mode="grayscale",
        shuffle=True,
    )

    train_nc_ds = train_nc_ds.map(lambda x: x / 255.0)

    print(f"Loading AD test data")
    test_ad_ds = tf.keras.utils.image_dataset_from_directory(
        path + "/test/AD",
        labels=None,
        label_mode=None,
        image_size=(60, 64),
        batch_size=None,
        seed=321,
        color_mode="grayscale",
        shuffle=True,
    )

    test_ad_ds = test_ad_ds.map(lambda x: x / 255.0)

    print(f"Loading NC test data")
    test_nc_ds = tf.keras.utils.image_dataset_from_directory(
        path + "/test/NC",
        labels=None,
        label_mode=None,
        image_size=(60, 64),
        batch_size=None,
        seed=321,
        color_mode="grayscale",
        shuffle=True,
    )

    test_nc_ds = test_nc_ds.map(lambda x: x / 255.0)

    print("Finished loading test data")

    train_both_ad_ds = tf.data.Dataset.zip(
        (
            train_ad_ds.shuffle(train_ad_ds.cardinality()),
            train_ad_ds.shuffle(train_ad_ds.cardinality()),
        )
    )
    train_both_nc_ds = tf.data.Dataset.zip(
        (
            train_nc_ds.shuffle(train_nc_ds.cardinality()),
            train_nc_ds.shuffle(train_nc_ds.cardinality()),
        )
    )
    train_diff_ds = tf.data.Dataset.zip(
        (
            train_ad_ds.shuffle(train_ad_ds.cardinality()),
            train_nc_ds.shuffle(train_nc_ds.cardinality()),
        )
    )

    train_ds = (
        train_both_ad_ds.concatenate(train_both_nc_ds)
        .concatenate(train_diff_ds)
        .shuffle(
            train_both_ad_ds.cardinality()
            + train_both_nc_ds.cardinality()
            + train_diff_ds.cardinality()
        )
    )
    to_take = round(0.2 * len(train_ds))
    validate_ds = train_ds.take(to_take)
    train_ds = train_ds.skip(to_take)

    test_both_ad_ds = tf.data.Dataset.zip(
        (
            test_ad_ds.shuffle(train_ad_ds.cardinality()),
            test_ad_ds.shuffle(train_ad_ds.cardinality()),
        )
    )
    test_both_nc_ds = tf.data.Dataset.zip(
        (
            test_nc_ds.shuffle(train_nc_ds.cardinality()),
            test_nc_ds.shuffle(train_nc_ds.cardinality()),
        )
    )
    test_diff_ds = tf.data.Dataset.zip(
        (
            test_ad_ds.shuffle(train_ad_ds.cardinality()),
            test_nc_ds.shuffle(train_nc_ds.cardinality()),
        )
    )

    test_ds = (
        test_both_ad_ds.concatenate(test_both_nc_ds)
        .concatenate(test_diff_ds)
        .shuffle(
            test_both_ad_ds.cardinality()
            + test_both_nc_ds.cardinality()
            + test_diff_ds.cardinality()
        )
    )

    print("Finished loading dataset")
    print("Train size:", len(train_ds))
    print("Validation size:", len(validate_ds))
    print("Test size:", len(test_ds))

    return train_ds.batch(32), validate_ds.batch(32), test_ds.batch(32)
