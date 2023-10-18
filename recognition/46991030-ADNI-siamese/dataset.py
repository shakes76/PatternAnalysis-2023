"""
dataset.py: Functions to load the dataset
"""
import numpy as np
import tensorflow as tf


@tf.function
def load_jpeg(path: str) -> tf.Tensor:
    """
    Loads a JPEG image from the given path and converts it to a tensor.

    The image is also standardised using `tf.image.per_image_standardization`.

    Args:
        path (str): The path to the JPEG image.

    Returns:
        tf.Tensor: The JPEG image as a tensor.
    """
    return tf.image.per_image_standardization(
        tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=1), dtype=tf.float32)
    )


def get_jpegs(path: str) -> list[str]:
    """
    Finds all the JPEG images in the given path.

    Args:
        path (str): The path to search for JPEG images.

    Returns:
        list[str]: A list of paths to JPEG images.
    """
    return tf.io.gfile.glob(f"{path}/*.jpeg")


def create_pairs(x1: list, x2: list, label: int) -> np.ndarray:
    """
    Creates pairs of images from the given lists of images.

    Args:
        x1 (list[str]): The first list of images.
        x2 (list[str]): The second list of images.
        label (int): The label to assign to the pairs.

    Returns:
        np.ndarray: An array of pairs of images, combined with the given label.
    """
    np.random.shuffle(x1)
    np.random.shuffle(x2)

    return np.array([(x1[i], x2[i], label) for i in range(min(len(x1), len(x2)))])


@tf.function
def map_ds_to_images(
    x1: tf.Tensor, x2: tf.Tensor, y: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Maps the given paths (as Tensors) to JPEG images.

    Args:
        x1 (tf.Tensor): The first path to map.
        x2 (tf.Tensor): The second path to map.
        y (tf.Tensor): The label to map.

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The first and second JPEG images, and the label.
    """
    return (load_jpeg(x1), load_jpeg(x2)), y


def load_dataset(
    path: str,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
]:
    """
    Loads the dataset from the given path.

    Args:
        path (str): The path to the dataset.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: The training, validation, and testing datasets for the SNN, and the training, validation, and testing datasets for the classifier.
    """
    print("Loading dataset")

    train_AD, train_NC = get_jpegs(f"{path}/train/AD"), get_jpegs(f"{path}/train/NC")

    train_both_AD, train_both_NC = create_pairs(train_AD, train_AD, 0), create_pairs(
        train_NC, train_NC, 0
    )

    train_mixed_AD, train_mixed_NC = create_pairs(train_AD, train_NC, 1), create_pairs(
        train_NC, train_AD, 1
    )

    train = np.concatenate(
        (train_both_AD, train_both_NC, train_mixed_AD, train_mixed_NC)
    )

    np.random.shuffle(train)

    minimum_length = min(len(train_AD), len(train_NC))

    classify_train = np.column_stack(
        (
            np.concatenate((train_AD[:minimum_length], train_NC[:minimum_length])),
            np.concatenate((np.zeros(minimum_length), np.ones(minimum_length))),
        )
    )
    np.random.shuffle(classify_train)

    test_AD, test_NC = get_jpegs(f"{path}/test/AD"), get_jpegs(f"{path}/test/NC")

    test_both_AD, test_both_NC = create_pairs(test_AD, test_AD, 0), create_pairs(
        test_NC, test_NC, 0
    )

    test_mixed_AD, test_mixed_NC = create_pairs(test_AD, test_NC, 1), create_pairs(
        test_NC, test_AD, 1
    )

    test = np.concatenate((test_both_AD, test_both_NC, test_mixed_AD, test_mixed_NC))

    minimum_length = min(len(test_AD), len(test_NC))

    classify_test = np.column_stack(
        (
            np.concatenate((test_AD[:minimum_length], test_NC[:minimum_length])),
            np.concatenate((np.zeros(minimum_length), np.ones(minimum_length))),
        )
    )

    validate = train[: len(train) // 5]
    train = train[len(train) // 5 :]

    classify_validate = classify_train[: len(classify_train) // 5]
    classify_train = classify_train[len(classify_train) // 5 :]

    # Load the dataset into tf.data.Dataset objects, map the paths to JPEG images, and batch them

    train_ds = (
        tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(train[:, 0]),
                tf.data.Dataset.from_tensor_slices(train[:, 1]),
                tf.data.Dataset.from_tensor_slices(train[:, 2].astype(np.float32)),
            )
        )
        .map(map_ds_to_images)
        .batch(32)
    )
    validate_ds = (
        tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(validate[:, 0]),
                tf.data.Dataset.from_tensor_slices(validate[:, 1]),
                tf.data.Dataset.from_tensor_slices(validate[:, 2].astype(np.float32)),
            )
        )
        .map(map_ds_to_images)
        .batch(32)
    )
    test_ds = (
        tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(test[:, 0]),
                tf.data.Dataset.from_tensor_slices(test[:, 1]),
                tf.data.Dataset.from_tensor_slices(test[:, 2].astype(np.float32)),
            )
        )
        .map(map_ds_to_images)
        .batch(32)
    )

    classify_train_ds = (
        tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(classify_train[:, 0]),
                tf.data.Dataset.from_tensor_slices(
                    classify_train[:, 1].astype(np.float32)
                ),
            ),
        )
        .map(lambda x1, x2: (load_jpeg(x1), x2))
        .batch(32)
    )

    classify_validate_ds = (
        tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(classify_validate[:, 0]),
                tf.data.Dataset.from_tensor_slices(
                    classify_validate[:, 1].astype(np.float32)
                ),
            ),
        )
        .map(lambda x1, x2: (load_jpeg(x1), x2))
        .batch(32)
    )

    classify_test_ds = (
        tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(classify_test[:, 0]),
                tf.data.Dataset.from_tensor_slices(
                    classify_test[:, 1].astype(np.float32)
                ),
            ),
        )
        .map(lambda x1, x2: (load_jpeg(x1), x2))
        .batch(32)
    )

    return (
        train_ds,
        validate_ds,
        test_ds,
        classify_train_ds,
        classify_validate_ds,
        classify_test_ds,
    )
