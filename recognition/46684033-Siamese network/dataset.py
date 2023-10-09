# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
import random

# Path for dataset
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

transform = transforms.Compose([
    transforms.ToTensor(),
])


def pair_dataset(dataset):
    # 0 is different class pair, 1 is same class pair
    pair_dataset = []
    AD_images = []
    NC_images = []
    try:
        pair_dataset = torch.load(r"C:\Users\wongm\Desktop\COMP3710\project\paired_images.pth")
        print("paired image list loaded")
    except FileNotFoundError as e:
        print("paired image list not found")

    if len(pair_dataset) == 0:
        for (image, label) in dataset:
            if label == 0:
                AD_images.append(image)
            else:
                NC_images.append(image)

        print("label splitted")
        # AD
        for i, image1 in enumerate(AD_images):
            # AD + AD
            idx = random.randint(0, len(AD_images) - 1)
            while idx == i:
                idx = random.randint(0, len(AD_images) - 1)
            image2 = AD_images[idx]
            pair_dataset.append(((image1, image2), 1))
            # AD+NC
            idx = random.randint(0, len(NC_images) - 1)
            image2 = NC_images[idx]
            pair_dataset.append(((image1, image2), 0))

        # NC
        for i, image1 in enumerate(NC_images):
            # NC+NC
            idx = random.randint(0, len(NC_images) - 1)
            while idx == i:
                idx = random.randint(0, len(NC_images) - 1)
            image2 = NC_images[idx]
            pair_dataset.append(((image1, image2), 1))
            # NC+AD
            idx = random.randint(0, len(AD_images) - 1)
            image2 = AD_images[idx]
            pair_dataset.append(((image1, image2), 0))
        torch.save(pair_dataset, r"C:\Users\wongm\Desktop\COMP3710\project\paired_images.pth")
    return pair_dataset


def load_data(train_path, test_path):
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    paired_trainset = pair_dataset(trainset)
    # paired_testset=pair_dataset(testset)

    train_loader = torch.utils.data.DataLoader(paired_trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64)
    return train_loader, test_loader


# for debug only
# train_loader, validation_loader, test_loader = load_data(train_path, test_path)
def load_data2(train_path, test_path):
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    paired_trainset = SiameseDatset(trainset)
    paired_testset = SiameseDatset(testset)
    train_loader = torch.utils.data.DataLoader(paired_trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(paired_testset, batch_size=64)
    return train_loader, test_loader
class SiameseDatset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __getitem__(self, idx):
        is_same_class = random.randint(0, 1)
        idx1 = random.randint(0, self.num_samples - 1)
        idx2 = random.randint(0, self.num_samples - 1)
        image1, label1 = self.dataset[idx1]
        image2, label2 = self.dataset[idx2]

        label = 0 if label1 == label2 else 1

        return (image1, image2), label

    def __len__(self):
        return self.num_samples
