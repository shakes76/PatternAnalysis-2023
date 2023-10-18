import numpy as np
import tensorflow as tf


@tf.function
def load_jpeg(path: str):
    return tf.image.per_image_standardization(
        tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=1), dtype=tf.float32)
    )


def get_jpegs(path: str) -> list[str]:
    return tf.io.gfile.glob(f"{path}/*.jpeg")


def create_pairs(x1, x2, label: int) -> np.ndarray:
    np.random.shuffle(x1)
    np.random.shuffle(x2)

    return np.array([(x1[i], x2[i], label) for i in range(min(len(x1), len(x2)))])


@tf.function
def map_ds_to_images(x1, x2, y):
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
    print("Loading dataset")

    train_AD, train_NC = get_jpegs(f"{path}/train/AD"), get_jpegs(f"{path}/train/NC")

    train_both_AD, train_both_NC = create_pairs(train_AD, train_AD, 0), create_pairs(
        train_NC, train_NC, 0
    )

    train_mixed = create_pairs(train_AD, train_NC, 1)

    train = np.concatenate(
        (train_both_AD, train_both_NC, train_mixed),
    )

    np.random.shuffle(train)

    classify_train = np.column_stack(
        (
            np.concatenate((train_AD, train_NC)),
            np.concatenate((np.zeros(len(train_AD)), np.ones(len(train_NC)))),
        )
    )
    np.random.shuffle(classify_train)

    test_AD, test_NC = get_jpegs(f"{path}/test/AD"), get_jpegs(f"{path}/test/NC")

    test_both_AD, test_both_NC = create_pairs(test_AD, test_AD, 0), create_pairs(
        test_NC, test_NC, 0
    )

    test_mixed = create_pairs(test_AD, test_NC, 1)

    test = np.concatenate((test_both_AD, test_both_NC, test_mixed))

    classify_test = np.column_stack(
        (
            np.concatenate((test_AD, test_NC)),
            np.concatenate((np.zeros(len(test_AD)), np.ones(len(test_NC)))),
        )
    )

    validate = train[: len(train) // 5]
    train = train[len(train) // 5 :]

    classify_validate = classify_train[: len(classify_train) // 5]
    classify_train = classify_train[len(classify_train) // 5 :]

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
