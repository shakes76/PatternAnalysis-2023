# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
import random
import os

#data augmentation
training_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomCrop(128, 16),
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

testing_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

batch_size = 16


def get_patient_number(file_path):
    """
    Get patient number based on the image path provided
    return patient number of the image
    """
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    return parts[0]


def load_data2(train_path, valid_path, test_path):
    """
    Load data from paths provided,
    return data loaders with paired image and labels
    """
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=training_transform)
    validationset = torchvision.datasets.ImageFolder(root=valid_path, transform=testing_transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=testing_transform)

    paired_trainset = SiameseDatset_contrastive(trainset)
    paired_validationset = SiameseDatset_contrastive(validationset)
    paired_testset = SiameseDatset_contrastive(testset)

    train_loader = torch.utils.data.DataLoader(paired_trainset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(paired_validationset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(paired_testset, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader


class SiameseDatset_contrastive(torch.utils.data.Dataset):
    """
    Pair images
    return a test image, positive image(same class with test image), negative image(different class with test image),
        1s, 0s, and class of test image
    """
    # same person label ==1, else label ==0
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __getitem__(self, idx1):
        idx2 = random.randint(0, self.num_samples - 1)
        idx3 = random.randint(0, self.num_samples - 1)
        image1, label1 = self.dataset[idx1]
        image2, label2 = self.dataset[idx2]
        image3, label3 = self.dataset[idx3]
        #Ensure it is a positive image and it is from a different patient
        while label1 != label2 and get_patient_number(self.dataset.imgs[idx1][0]) != get_patient_number(
                self.dataset.imgs[idx2][0]):
            idx2 = random.randint(0, self.num_samples - 1)
            image2, label2 = self.dataset[idx2]
        #Ensure it is a negative image
        while label1 == label3:
            idx3 = random.randint(0, self.num_samples - 1)
            image3, label3 = self.dataset[idx3]
        labela = 1
        labelb = 0

        return (image1, image2, image3), labela, labelb, label1

    def __len__(self):
        return self.num_samples