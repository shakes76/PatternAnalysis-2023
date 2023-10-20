from parameters import *
from modules import *
from torchvision import datasets, transforms as T
import numpy as np
import matplotlib.pyplot as plt


def load_data_celeba():
    # Defining Transforms
    transform_train = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_sd),
        T.RandomHorizontalFlip(),
        # T.RandomCrop(32, padding=4, padding_mode='reflect'),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(norm_mean, norm_sd),
    ])

    transform = T.ToPILImage()

    # Obtaining datasets
    train_dataset = datasets.CelebA(root, "train", transform=transform_train, download=True)  # Set batch to = 10, or 82

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, transform, train_dataset


def load_brain_images():
    dataset = np.empty((3939, 1, 1, 256, 256))
    data_len = 42
    k = 0
    # Loading in slices of MRI scan as images for training data
    for i in range(1, data_len + 1):
        # Handling inconsistent data
        missing_list = [8, 24, 36]
        if i in missing_list:
            continue

        n3_list = [7, 15, 16, 20, 26, 34, 38, 39]
        if i in n3_list:
            n = "n3"
        else:
            n = "n4"
        prepend = "" if i >= 10 else "0"
        numerator = "00" + prepend + str(i)
        filepath = "oasis_cross-sectional_disc1\disc1\OAS1_" + numerator + "_MR1\PROCESSED\MPRAGE\SUBJ_111\OAS1_" + numerator + "_MR1_mpr_" + n + "_anon_sbj_111.img"
        fid = open(filepath, 'rb')
        data = np.fromfile(fid, np.dtype('>u2'))
        image = data.reshape((160, 256, 256))
        for j, img in enumerate(image):
            # Narrowing down the scans showing images of the skull and not brain scans
            if j >= 30 and j <= 130:
                dataset[k][0][0] = img
                k += 1

    return dataset
