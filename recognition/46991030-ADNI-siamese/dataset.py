import tensorflow as tf


def load_dataset(path: str) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the dataset from the given path
    :param path: Path to the dataset
    :return: train_ds, val_ds, test_ds
    """
    train_data_path = path + "/train"
    test_data_path = path + "/test"

    print(f"Loading training data from {train_data_path}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        validation_split=0.2,
        subset="training",
        image_size=(64, 64),
        batch_size=32,
        seed=321,
        color_mode="grayscale",
    )

    print(f"Loading validation data from {train_data_path}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        validation_split=0.2,
        subset="validation",
        image_size=(64, 64),
        batch_size=32,
        seed=321,
        color_mode="grayscale",
    )

    print(f"Loading test data from {test_data_path}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_path, image_size=(64, 64), batch_size=32, color_mode="grayscale"
    )

    print("Finished loading dataset")

    return train_ds, val_ds, test_ds
