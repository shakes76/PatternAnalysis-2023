import tensorflow as tf


@tf.function
def load_jpeg(path: str):
    return tf.cast(
        tf.io.decode_jpeg(tf.io.read_file(path), channels=1), dtype=tf.float32
    ) / tf.constant(255.0)


def get_jpegs_at_path(path: str) -> tf.data.Dataset:
    return tf.data.Dataset.list_files(f"{path}/*.jpeg", shuffle=True)


def create_pairs(x1: tf.data.Dataset, x2: tf.data.Dataset, label):
    return tf.data.Dataset.zip(
        (x1.shuffle(x1.cardinality()), x2.shuffle(x2.cardinality()))
    ).map(lambda x1, x2: (x1, x2, tf.constant(label, dtype=tf.float32)))


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

    print("Finding training data")

    train_AD = get_jpegs_at_path(f"{path}/train/AD")
    train_NC = get_jpegs_at_path(f"{path}/train/NC")

    print("Creating pairs")

    train_both_AD = create_pairs(train_AD, train_AD, 0)
    train_both_NC = create_pairs(train_NC, train_NC, 0)
    train_mixed_1 = create_pairs(train_AD, train_NC, 1)
    train_mixed_2 = create_pairs(train_NC, train_AD, 1)

    print("Shuffling")

    train_ds = (
        train_both_AD.concatenate(
            train_both_NC,
        )
        .concatenate(
            train_mixed_1,
        )
        .concatenate(
            train_mixed_2,
        )
        .shuffle(
            train_both_AD.cardinality()
            + train_both_NC.cardinality()
            + train_mixed_1.cardinality()
            + train_mixed_2.cardinality()
        )
    )

    train_AD = train_AD.map(lambda x: (x, tf.constant(0, dtype=tf.float32)))
    train_NC = train_NC.map(lambda x: (x, tf.constant(1, dtype=tf.float32)))
    class_train_ds = tf.data.Dataset.concatenate(train_AD, train_NC).shuffle(
        train_AD.cardinality() + train_NC.cardinality()
    )

    print("Finding testing data")

    test_AD = get_jpegs_at_path(f"{path}/test/AD")
    test_NC = get_jpegs_at_path(f"{path}/test/NC")

    print("Creating pairs")

    test_both_AD = create_pairs(test_AD, test_AD, 0)
    test_both_NC = create_pairs(test_NC, test_NC, 0)
    test_mixed_1 = create_pairs(test_AD, test_NC, 1)
    test_mixed_2 = create_pairs(test_NC, test_AD, 1)

    print("Shuffling")

    test_ds = (
        test_both_AD.concatenate(
            test_both_NC,
        )
        .concatenate(
            test_mixed_1,
        )
        .concatenate(
            test_mixed_2,
        )
        .shuffle(
            test_both_AD.cardinality()
            + test_both_NC.cardinality()
            + test_mixed_1.cardinality()
            + test_mixed_2.cardinality()
        )
    )

    test_AD = test_AD.map(lambda x: (x, tf.constant(0, dtype=tf.float32)))
    test_NC = test_NC.map(lambda x: (x, tf.constant(1, dtype=tf.float32)))
    class_test_ds = tf.data.Dataset.concatenate(test_AD, test_NC).shuffle(
        test_AD.cardinality() + test_NC.cardinality()
    )

    print("Loading images")
    train_ds = train_ds.map(lambda x1, x2, y: ((load_jpeg(x1), load_jpeg(x2)), y))
    test_ds = test_ds.map(lambda x1, x2, y: ((load_jpeg(x1), load_jpeg(x2)), y))
    class_train_ds = class_train_ds.map(lambda x, y: (load_jpeg(x), y))
    class_test_ds = class_test_ds.map(lambda x, y: (load_jpeg(x), y))

    validate_ds = train_ds.shuffle(train_ds.cardinality()).take(
        train_ds.cardinality() // 5
    )
    train_ds = train_ds.shuffle(train_ds.cardinality()).skip(
        train_ds.cardinality() // 5
    )

    class_validate_ds = class_train_ds.take(class_train_ds.cardinality() // 5)
    class_train_ds = class_train_ds.skip(class_train_ds.cardinality() // 5)

    return (
        train_ds.batch(32),
        validate_ds.batch(32),
        test_ds.batch(32),
        class_train_ds.batch(32),
        class_validate_ds.batch(32),
        class_test_ds.batch(32),
    )
