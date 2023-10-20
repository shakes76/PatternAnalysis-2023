# helper functions
import random
import glob
import shutil
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

def show_image(data: tuple):
    """Shows the image of a sample from a dataset

    Args:
        data (tuple): tuple of PIL image, label
    """
    image = data[0]
    label = "AD" if data[1] == 0 else "NC"
    plt.text(25, 50, label, color="white")

    plt.axis("off")
    plt.imshow(image)
    plt.show()


def train_valid_split(data_path="AD_NC/", new_path="data/", verbose=False):
    if verbose:
        print("Performing train-validation split...")
    test_data_original = glob.glob("*/*.jpeg", root_dir=data_path + "train/")

    # group the data by patient id
    grouped_data = {}
    for img_path in test_data_original:
        img_class = img_path[:2]
        if img_class not in grouped_data.keys():
            grouped_data[img_class] = {}

        img_id = img_path[3:-5]
        patient, _ = img_id.split("_")
        if patient not in grouped_data[img_path[:2]].keys():
            grouped_data[img_class][patient] = []

        grouped_data[img_class][patient].append(img_path[3:])

    # 80-20 split on both classes
    for img_class in grouped_data.keys():
        patients = list(grouped_data[img_class].keys())
        random.shuffle(patients)
        approx_80 = int(len(patients) * 0.8)

        train_path = new_path + f"train/{img_class}"
        valid_path = new_path + f"valid/{img_class}"
        for dir_path in [train_path, valid_path]:
            dest_dir = Path(dir_path)
            if dest_dir.is_dir():
                shutil.rmtree(dest_dir)
            os.makedirs(dir_path, exist_ok=True)

        dest_path = train_path
        for i, patient in enumerate(patients):
            if i >= approx_80:
                dest_path = valid_path

            for img_path in grouped_data[img_class][patient]:
                shutil.copy(f"{data_path}train/{img_class}/{img_path}",
                            dest_path)
            if verbose:
                print(f"{dest_path}/{img_path} done...")

    # copy test directory over to new path
    shutil.copytree(f"{data_path}test/", new_path + "test/", dirs_exist_ok=True)
    if verbose:
        print("Copied test data...")

def plot_losses(train_losses, valid_losses):
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    ax[1].plot(range(1, len(valid_losses) + 1), valid_losses, label="Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("Triplet Loss")
    ax[1].set_ylabel("Triplet Loss")
    ax[0].legend()
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./results/losses_{datetime.datetime.now().strftime('%H-%M-%S')}")

def plot_test_loss(test_losses):
    fig, ax = plt.subplots()
    ax.hist(test_losses, bins=25, color="r")
    ax.set_title("Loss over test dataset")
    ax.set_xlabel("Testing loss")
    ax.set_xlabel("Count of Samples")
    plt.tight_layout()
    plt.savefig(f"./results/test_loss_{datetime.datetime.now().strftime('%H-%M-%S')}")

