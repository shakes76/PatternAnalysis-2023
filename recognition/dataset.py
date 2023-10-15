## Data Set

import os
from glob import glob
from sklearn.model_selection import train_test_split


H = 256
W = 256


def load_data(dataset_path, split=0.2):
    images = sorted(glob(os.path.join(dataset_path, "ISIC-2017_Training_Data", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "ISIC-2017_Training_Part1_GroundTruth", "*.png")))

    test_size = int(len(images) * split)

    # Split into 60/20/20
    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


if __name__ == "__main__":
    dataset_path = r"C:\Users\raulm\Desktop\Uni\Sem2.2023\Patterns\ISIC-2017_Training_Data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    