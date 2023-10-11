import numpy as np
from tensorflow import keras
import os


def load_jpegs_at_path(path: str) -> list[np.ndarray]:
    """
    Loads all jpeg images from the given path
    :param path: Path to the directory containing the images
    :return: List of image data
    """

    return [
        keras.utils.img_to_array(
            keras.utils.load_img(
                f"{path}/{p}", target_size=(60, 64), color_mode="grayscale"
            )
        )
        / 255.0
        for p in os.listdir(path)
        if p.endswith(".jpeg")
    ]


def create_pairs(x1, x2, label):
    np.random.shuffle(x1)
    np.random.shuffle(x2)

    return [((x1[i], x2[i]), label) for i in range(min(len(x1), len(x2)))]


def load_dataset(
    path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Loads the ADNI dataset from the given path
    :param path: Path to the dataset
    :return train_X1, train_X2, train_y, validate_X1, validate_X2, validate_y, test_X1, test_X2, test_y
    """
    print("Loading dataset")

    print("Loading training data")

    train_AD = load_jpegs_at_path(f"{path}/train/AD")
    train_NC = load_jpegs_at_path(f"{path}/train/NC")

    print("Creating pairs")

    train_both_AD = create_pairs(train_AD, train_AD, 0)
    train_both_NC = create_pairs(train_NC, train_NC, 0)
    train_mixed_1 = create_pairs(train_AD, train_NC, 1)
    train_mixed_2 = create_pairs(train_NC, train_AD, 1)

    print("Shuffling")

    train = train_both_AD + train_both_NC + train_mixed_1 + train_mixed_2
    np.random.shuffle(train)

    print("Creating numpy arrays")
    train_X = np.array([t[0] for t in train])
    train_y = np.array([t[1] for t in train])

    print("Loading testing data")

    test_AD = load_jpegs_at_path(f"{path}/test/AD")
    test_NC = load_jpegs_at_path(f"{path}/test/NC")

    print("Creating pairs")

    test_both_AD = create_pairs(test_AD, test_AD, 0)
    test_both_NC = create_pairs(test_NC, test_NC, 0)
    test_mixed_1 = create_pairs(test_AD, test_NC, 1)
    test_mixed_2 = create_pairs(test_NC, test_AD, 1)

    print("Shuffling")

    test = test_both_AD + test_both_NC + test_mixed_1 + test_mixed_2
    np.random.shuffle(test)

    test_X = np.array([t[0] for t in test])
    test_y = np.array([t[1] for t in test])

    num_validate = len(train_X) // 5  # 20% split

    validate_X = train_X[:num_validate]
    validate_y = train_y[:num_validate]

    train_X = train_X[num_validate:]
    train_y = train_y[num_validate:]

    return (
        train_X[:, 0],
        train_X[:, 1],
        train_y.astype(np.float32),
        validate_X[:, 0],
        validate_X[:, 1],
        validate_y.astype(np.float32),
        test_X[:, 0],
        test_X[:, 1],
        test_y.astype(np.float32),
    )
