import torchvision
import torch
import torchvision.transforms as transforms
import random
import utils

TRAIN_PATH = utils.train_path

TEST_PATH = utils.test_path

class ClassifierDataset(torch.utils.data.Dataset):
    """
    Custom dataset for the classifier model.

    Attributes:
        trainset list(torch.utils.data.Dataset): The list containing the dataset
    """
    def __init__(self, trainset):
        self.trainset = trainset

    def __getitem__(self, index):
        img, label = self.trainset[index]
        return img, label
    
    def __len__(self):
        return len(self.trainset)

class PairedDataset(torch.utils.data.Dataset):
    """
        Custom dataset for the siamese model.
        # follow this source: https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/

        Attributes:
            trainset list(torch.utils.data.Dataset): The list containing the dataset
    """
    def __init__(self, trainset):
        random.seed(42)
        self.trainset = trainset

    def __getitem__(self, index):
        img0, label0 = self.trainset[index]

        # randomly chosen whether the pair have the same label or not (both AD/NC or one each)
        check_same_class = random.randint(0,1) 
        
        while True:
            img1, label1 = random.choice(self.trainset)
            
            if not torch.equal(img0, img1): # ignore image that are identical
                if check_same_class and label0 == label1:
                    break
                elif not check_same_class and label0 != label1:
                    break
        
        return img0, img1, check_same_class
    
    def __len__(self):
        return len(self.trainset)

class LoadData():
    """
        Handle loading dataset
    """
    def __init__(self, train=True, siamese=True):
        # seed for reproduce the run -> seed at init so every other method is impact by the seed
        torch.manual_seed(42)
        random.seed(42)
        self.train = train
        self.siamese = siamese

        # data loader hyper parameter
        self.image_size = 224
        self.batch_size = 64
        self.num_worker = 0

        # train and validate ratio
        self.train_ratio = 0.8
        self.val_ratio = 1 - self.train_ratio

    def split_dataset(self, dataset):
        """
            split data set into validate and train group
            split based on patient-wise

            Attribute:
                dataset (torch.utils.data.Dataset): ADMI dataset
            
            Return:
                Tuple(List, List): train_list and validate_list that is proportion to the train_ratio and validate_ratio respectively
        """
        patient_dict_ad = dict()
        patient_dict_nc = dict()

        current_patient = dataset.imgs[0][0].split("/")[-1].split("_")[0]
        current_label = dataset.imgs[0][1]
        count = 0
        # have two dictionary of patient where a patient can have multiple image
        for img, label in dataset.imgs:
            next_patient = img.split("/")[-1].split("_")[0]
            if next_patient == current_patient:
                count += 1
            else:
                if current_label == 0:
                    patient_dict_ad[current_patient] = count
                else:
                    patient_dict_nc[current_patient] = count
                current_patient = next_patient
                current_label = label
                count = 1
        
        # append the last patient -> the logic above left out the last patient
        if current_label == 0:
            patient_dict_ad[current_patient] = count
        else:
            patient_dict_nc[current_patient] = count

        # split to val and train list
        val_set = set()
        train_set = set()

        # split the patient to its respective group (val or train) -> set is more efficient for lookup time compare to list
        self.split_patient_name(val_set, train_set, patient_dict_ad)
        self.split_patient_name(val_set, train_set, patient_dict_nc)

        train_list = []
        val_list = []

        for i, (img, _) in enumerate(dataset.imgs):
            patient = img.split("/")[-1].split("_")[0]

            if patient in val_set: # append the real data into the respective list
                val_list.append(dataset[i])
            else:
                train_list.append(dataset[i])
        
        return train_list, val_list
            
    def split_patient_name(self, val_set, train_set, patient_dict):
        """
            Split the patient name to its respective set

            Attribute:
                val_set (set): validate set
                train_set (set): train set
                patient_dict (dict): dictionary contain all patient id
        """
        patients = list(patient_dict.keys())
        random.shuffle(patients) # shuffle the patients key -> random

        total_count_patient = sum(patient_dict[patient] for patient in patients)
        val_portion = total_count_patient * self.val_ratio

        count = 0
        for patient in patients:
            if count <= val_portion: # update set in place -> add the respective patient to the corresponding set
                val_set.add(patient)
            else:
                train_set.add(patient)
            count += patient_dict[patient]
        
    def load_data(self):
        """
            Handle dataloader train, validate and siamese
            the loader is dependant on the init value

            Transformation adopt from this https://github.com/metinmertakcay/video-classification-cnn-lstm/blob/main/Classify_Frame_with_ResNet18_UCFSport.ipynb

            Return
                data (torch.utils.data.Data): respective dataset -> either train, validate pair or test dataset
        """
        if self.train: # transforms parameter for train/set dataset
            path = TRAIN_PATH
            transform = transforms.Compose([
                transforms.RandomCrop(self.image_size, 15, padding_mode='reflect'),
                transforms.ToTensor(),
            ])
        else:
            path = TEST_PATH
            transform = transforms.Compose([
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ])

        image = torchvision.datasets.ImageFolder(root=path, transform=transform)

        # split the data into validate and train image
        train_image, val_image = self.split_dataset(image)
        if self.train:
            if self.siamese:
                # Use custom pair data set to handle siamese model train
                trainset = PairedDataset(train_image)
                valset = PairedDataset(val_image)

                train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                val_loader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)

                return train_loader, val_loader
            else:
                # use custom classifier dataset to handle binary classifier model train
                trainset = ClassifierDataset(train_image)
                valset = ClassifierDataset(val_image)

                train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                val_loader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)

                return train_loader, val_loader
        else:
            # use default data loader for test loader because not need to split test dataset for validate
            test_loader = torch.utils.data.DataLoader(image, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
            return test_loader

if __name__ == '__main__':

    # test dataset
    print("Start test load data")
    load_train_siamese, load_val_siamese = LoadData(train=True, siamese=True).load_data()
    load_train_classifier, load_val_classifier = LoadData(train=True, siamese=False).load_data()
    load_test = LoadData(train=False).load_data()
    print("Finish test load data")

    