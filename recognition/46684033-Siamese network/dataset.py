# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
import random

# Path for dataset
train_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/train"
test_path = r"C:/Users/wongm/Downloads/ADNI_AD_NC_2D/AD_NC/test"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),


])


def load_data2(train_path, test_path):
    dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    trainset, validation_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    paired_trainset = SiameseDatset_contrastive(trainset)
    paired_validationset = SiameseDatset_contrastive(validation_set)
    #paired_testset = SiameseDatset_contrastive(testset)
    train_loader = torch.utils.data.DataLoader(paired_trainset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(paired_validationset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return train_loader,validation_loader,test_loader
class SiameseDatset_BCE(torch.utils.data.Dataset):
    #same person label ==1, else label ==0
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)
    def __getitem__(self, idx1):
        idx2 = random.randint(0, self.num_samples - 1)
        image1, label1 = self.dataset[idx1]
        image2, label2 = self.dataset[idx2]
        if label1 == label2:
            label = 1
        else:
            label = 0
        return (image1, image2), label
    def __len__(self):
        return self.num_samples

class SiameseDatset_contrastive(torch.utils.data.Dataset):
    #same person label ==1, else label ==0
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __getitem__(self, idx1):
        is_same_class = random.randint(0, 1)

        idx2 = random.randint(0, self.num_samples - 1)
        idx3 = random.randint(0, self.num_samples - 1)
        image1, label1 = self.dataset[idx1]
        image2, label2 = self.dataset[idx2]
        image3, label3 = self.dataset[idx3]
        while label1 != label2:
            idx2 = random.randint(0, self.num_samples - 1)
            image2, label2 = self.dataset[idx2]
        while label1 == label3:
            idx3 = random.randint(0, self.num_samples - 1)
            image3, label3 = self.dataset[idx3]
        # if label1 == label2:
        #     label = 1
        # else:
        #     label = 0
        labela = 1
        labelb = 0

        return (image1, image2, image3), labela,labelb,label1

    def __len__(self):
        return self.num_samples

def load_data3(train_path, test_path):
    dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    trainset, validation_set = torch.utils.data.random_split(dataset, [train_size, val_size])


    paired_trainset = SiameseDatset_triplet(trainset)
    paired_validation_set = SiameseDatset_triplet(testset)

    train_loader = torch.utils.data.DataLoader(paired_trainset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(paired_validation_set, batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return train_loader,validation_loader,test_loader

class SiameseDatset_triplet(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __getitem__(self, idx1):
        is_same_class = random.randint(0, 1)
        idx2 = random.randint(0, self.num_samples - 1)
        idx3 = random.randint(0, self.num_samples - 1)
        anchor_image, label1 = self.dataset[idx1]
        #get positive image
        pos_image, label2 = self.dataset[idx2]
        while(label1 != label2 or idx1==idx2):
            idx2 = random.randint(0, self.num_samples - 1)
            pos_image, label2 = self.dataset[idx2]
        #get negative image
        neg_image, label3 = self.dataset[idx3]
        while (label1 == label3):
            idx3 = random.randint(0, self.num_samples - 1)
            neg_image, label3 = self.dataset[idx3]

        return anchor_image, pos_image, neg_image

    def __len__(self):
        return self.num_samples

class SiameseDatset_test(torch.utils.data.Dataset):
    def __init__(self, trainset,testset):
        self.trainset = trainset
        self.testset = testset
        self.num_samples_train = len(trainset)
        self.num_samples_test = len(testset)

    def __getitem__(self, idx):
        idx = random.randint(0, self.num_samples_test - 1)
        pos_idx1 = random.randint(0, self.num_samples_train - 1)
        pos_idx2 = random.randint(0, self.num_samples_train - 1)
        pos_idx3 = random.randint(0, self.num_samples_train - 1)
        pos_idx4 = random.randint(0, self.num_samples_train - 1)
        pos_idx5 = random.randint(0, self.num_samples_train - 1)
        pos_idx6 = random.randint(0, self.num_samples_train - 1)
        pos_idx7 = random.randint(0, self.num_samples_train - 1)
        pos_idx8 = random.randint(0, self.num_samples_train - 1)
        pos_idx9 = random.randint(0, self.num_samples_train - 1)
        pos_idx10 = random.randint(0, self.num_samples_train - 1)
        neg_idx1 = random.randint(0, self.num_samples_train - 1)
        neg_idx2 = random.randint(0, self.num_samples_train - 1)
        neg_idx3 = random.randint(0, self.num_samples_train - 1)
        neg_idx4 = random.randint(0, self.num_samples_train - 1)
        neg_idx5 = random.randint(0, self.num_samples_train - 1)
        neg_idx6 = random.randint(0, self.num_samples_train - 1)
        neg_idx7 = random.randint(0, self.num_samples_train - 1)
        neg_idx8 = random.randint(0, self.num_samples_train - 1)
        neg_idx9 = random.randint(0, self.num_samples_train - 1)
        neg_idx10 = random.randint(0, self.num_samples_train - 1)
        test_image, label1 = self.testset[idx]

        #get positive image
        pos_image1, label2 = self.trainset[pos_idx1]
        while(label1 != label2):
            pos_idx1 = random.randint(0, self.num_samples_train - 1)
            pos_image1, label2 = self.trainset[pos_idx1]

        pos_image2, label2 = self.trainset[pos_idx2]
        while (label1 != label2):
            pos_idx2 = random.randint(0, self.num_samples_train - 1)
            pos_image2, label2 = self.trainset[pos_idx2]

        pos_image3, label2 = self.trainset[pos_idx3]
        while (label1 != label2):
            pos_idx3 = random.randint(0, self.num_samples_train - 1)
            pos_image3, label2 = self.trainset[pos_idx3]

        pos_image4, label2 = self.trainset[pos_idx4]
        while (label1 != label2):
            pos_idx4 = random.randint(0, self.num_samples_train - 1)
            pos_image4, label2 = self.trainset[pos_idx4]

        pos_image5, label2 = self.trainset[pos_idx5]
        while (label1 != label2):
            pos_idx5 = random.randint(0, self.num_samples_train - 1)
            pos_image5, label2 = self.trainset[pos_idx5]

        pos_image6, label2 = self.trainset[pos_idx6]
        while (label1 != label2):
            pos_idx6 = random.randint(0, self.num_samples_train - 1)
            pos_image6, label2 = self.trainset[pos_idx6]

        pos_image7, label2 = self.trainset[pos_idx7]
        while (label1 != label2):
            pos_idx7 = random.randint(0, self.num_samples_train - 1)
            pos_image7, label2 = self.trainset[pos_idx7]

        pos_image8, label2 = self.trainset[pos_idx8]
        while (label1 != label2):
            pos_idx8 = random.randint(0, self.num_samples_train - 1)
            pos_image8, label2 = self.trainset[pos_idx8]

        pos_image9, label2 = self.trainset[pos_idx9]
        while (label1 != label2):
            pos_idx9 = random.randint(0, self.num_samples_train - 1)
            pos_image9, label2 = self.trainset[pos_idx9]

        pos_image10, label2 = self.trainset[pos_idx10]
        while (label1 != label2):
            pos_idx10 = random.randint(0, self.num_samples_train - 1)
            pos_image10, label2 = self.trainset[pos_idx10]

        #get negative image
        neg_image1, label3 = self.trainset[neg_idx1]
        while (label1 == label3):
            neg_idx1 = random.randint(0, self.num_samples_train - 1)
            neg_image1, label3 = self.trainset[neg_idx1]

        neg_image2, label3 = self.trainset[neg_idx2]
        while (label1 == label3):
            neg_idx2 = random.randint(0, self.num_samples_train - 1)
            neg_image2, label3 = self.trainset[neg_idx2]

        neg_image3, label3 = self.trainset[neg_idx3]
        while (label1 == label3):
            neg_idx3 = random.randint(0, self.num_samples_train - 1)
            neg_image3, label3 = self.trainset[neg_idx3]

        neg_image4, label3 = self.trainset[neg_idx4]
        while (label1 == label3):
            neg_idx4 = random.randint(0, self.num_samples_train - 1)
            neg_image4, label3 = self.trainset[neg_idx4]

        neg_image5, label3 = self.trainset[neg_idx5]
        while (label1 == label3):
            neg_idx5 = random.randint(0, self.num_samples_train - 1)
            neg_image5, label3 = self.trainset[neg_idx5]

        neg_image6, label3 = self.trainset[neg_idx6]
        while (label1 == label3):
            neg_idx6 = random.randint(0, self.num_samples_train - 1)
            neg_image6, label3 = self.trainset[neg_idx6]

        neg_image7, label3 = self.trainset[neg_idx7]
        while (label1 == label3):
            neg_idx7 = random.randint(0, self.num_samples_train - 1)
            neg_image7, label3 = self.trainset[neg_idx7]

        neg_image8, label3 = self.trainset[neg_idx8]
        while (label1 == label3):
            neg_idx8 = random.randint(0, self.num_samples_train - 1)
            neg_image8, label3 = self.trainset[neg_idx8]

        neg_image9, label3 = self.trainset[neg_idx9]
        while (label1 == label3):
            neg_idx9 = random.randint(0, self.num_samples_train - 1)
            neg_image9, label3 = self.trainset[neg_idx9]

        neg_image10, label3 = self.trainset[neg_idx10]
        while (label1 == label3):
            neg_idx10 = random.randint(0, self.num_samples_train - 1)
            neg_image10, label3 = self.trainset[neg_idx10]

        return test_image, pos_image1,pos_image2,pos_image3,pos_image4,pos_image5,pos_image6,pos_image7,pos_image8,pos_image9,pos_image10, neg_image1,neg_image2,neg_image3,neg_image4, neg_image5,neg_image6,neg_image7,neg_image8,neg_image9,neg_image10, label1

    def __len__(self):
        return self.num_samples_test