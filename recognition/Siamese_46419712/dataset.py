import torchvision
import torch
import torchvision.transforms as transforms
import random
import numpy as np

TRAIN_PATH = "/home/groups/comp3710/ADNI/AD_NC/train"
TRAIN_PATH = "./AD_NC/train"

TEST_PATH = "/home/groups/comp3710/ADNI/AD_NC/train"
TEST_PATH = "./AD_NC/train"

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, trainset):
        self.trainset = trainset

    def __getitem__(self, index):
        img, label = self.trainset[index]
        return img, label
    
    def __len__(self):
        return len(self.trainset)

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, trainset):
        # follow this source: https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
        self.trainset = trainset

    def __getitem__(self, index):
        img0, label0 = self.trainset[index]

        check_same_class = random.randint(0,1) 
        
        while True:
            img1, label1 = random.choice(self.trainset)

            if not torch.equal(img0, img1):
                if check_same_class and label0 == label1:
                    break
                elif not check_same_class and label0 != label1:
                    break
        
        return img0, img1, check_same_class
    
    def __len__(self):
        return len(self.trainset)

class LoadData():
    def __init__(self, train=True, siamese=True):
        torch.manual_seed(35)
        random.seed(35)
        self.train = train
        self.siamese = siamese

        self.image_size = 105
        self.batch_size = 128
        self.num_worker = 0

        self.train_ratio = 0.9
        self.val_ratio = 1 - self.train_ratio

    def split_dataset1(self, dataset, seed=True):
        patient_dict_ad = dict()
        patient_dict_nc = dict()

        current_patient = dataset.imgs[0][0].split("/")[-1].split("_")[0]
        current_label = dataset.imgs[0][1]
        count = 0
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
        
        # append the last patient
        if current_label == 0:
            patient_dict_ad[current_patient] = count
        else:
            patient_dict_nc[current_patient] = count

        # split to val and train list
        val_set = set()
        train_set = set()

        self.split_patient_name(val_set, train_set, patient_dict_ad)
        self.split_patient_name(val_set, train_set, patient_dict_nc)


        train_list = []
        val_list = []

        for i, (img, _) in enumerate(dataset.imgs):
            patient = img.split("/")[-1].split("_")[0]

            if patient in val_set:
                val_list.append(dataset[i])
            else:
                train_list.append(dataset[i])
        
        return train_list, val_list
            
    def split_patient_name(self, val_set, train_set, patient_dict):
        patients = list(patient_dict.keys())
        random.shuffle(patients) # shuffle the patients key -> random

        total_count_patient = sum(patient_dict[patient] for patient in patients)
        val_portion = total_count_patient * self.val_ratio

        count = 0
        for patient in patients:
            if count <= val_portion:
                val_set.add(patient)
            else:
                train_set.add(patient)
            count += patient_dict[patient]
        
    def split_dataset(self, dataset, seed=True):
        # if seed:
        #     generator = torch.Generator().manual_seed(35)
        # else:
        #     generator = torch.Generator()

        # # follow the principle -> 80% train 20% val
        # train_set, val_set = torch.utils.data.random_split(dataset, [self.train_ratio, self.val_ratio], generator=generator)

        # print(dataset.imgs[0: 100])
        current_label = 0
        for i, (img, label) in enumerate(dataset.imgs):
            if i == 0:
                current_label = label
            
            if label != current_label:
                break
        
        # print(i)
        len_AD = i

        len_NC = len(dataset) - len_AD
        # print(dataset[len_AD - 1]) # first AD
        # print(dataset[len_AD + len_NC - 1]) # first NC
        AD_list = []
        NC_list = []

        index_AD = 1
        current_patient = dataset.imgs[0][0].split("/")[-1].split("_")[0]
        index_list = [dataset[0]]
        while index_AD < len_AD:
            next_patient = dataset.imgs[index_AD][0].split("/")[-1].split("_")[0]
            if current_patient != next_patient:
                AD_list.append(index_list)
                index_list = [dataset[index_AD]]
                current_patient = next_patient
            else:
                index_list.append(dataset[index_AD])
            index_AD += 1
        AD_list.append(index_list)

        # at this stage, AD_list contain all the AD_images, in the form of [[] <- the patient]
        # return train_set, val_set
        
        index_NC = len_AD + 1
        current_patient = dataset.imgs[index_NC - 1][0].split("/")[-1].split("_")[0]
        index_list = [dataset[index_NC - 1]]
        while index_NC < len(dataset):
            next_patient = dataset.imgs[index_NC][0].split("/")[-1].split("_")[0]
            if current_patient != next_patient:
                NC_list.append(index_list)
                index_list = [dataset[index_NC]]
                current_patient = next_patient
            else:
                index_list.append(dataset[index_NC])
            index_NC += 1
        NC_list.append(index_list)

        val_list = []

        train_list = []

        while len(val_list) < self.val_ratio * len_AD:
            index = random.randint(0, len(AD_list))
            patient = AD_list.pop(index)
            val_list.extend(patient)

        for val in AD_list:
            train_list.extend(val)
        
        new_val_list_len = len(val_list)

        while len(val_list) - new_val_list_len < self.val_ratio * len_NC:
            index = random.randint(0, len(NC_list))
            patient = NC_list.pop(index)
            val_list.extend(patient)

        for val in NC_list:
            train_list.extend(val)

        return train_list, val_list
        # print(len(val_list))
        # print(len(train_list))
        # print("finish")

    def load_data(self):
        if self.train:
            path = TRAIN_PATH
            transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])
        else:
            path = TEST_PATH
            transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

        image = torchvision.datasets.ImageFolder(root=path, transform=transform)
        train_image, val_image = self.split_dataset(image)
        if self.train:
            if self.siamese:
                trainset = PairedDataset(train_image)
                # print(trainset)
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)

                valset = PairedDataset(val_image)
                val_loader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                return train_loader, val_loader
            else:
                trainset = ClassifierDataset(train_image)

                valset = ClassifierDataset(val_image)
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                val_loader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                return train_loader, val_loader
        else:
            test_loader = torch.utils.data.DataLoader(image, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
            return test_loader

if __name__ == '__main__':

    TRAIN_PATH = "./AD_NC/train"
    transform = transforms.Compose([
        transforms.Resize(105),
        transforms.CenterCrop(105),
        transforms.ToTensor()
    ])
    image = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
    # print(len(image) * 0.9)
    load_data = LoadData(siamese=False).split_dataset1(image)

    

    # load_data.split_dataset(image)

    # test dataset
    # print("Start test load data")
    # load_train_siamese, load_val_siamese = LoadData(train=True, siamese=True).load_data()
    # load_train_classifier, load_val_classifier = LoadData(train=True, siamese=False).load_data()
    # load_test = LoadData(train=False).load_data()
    # print("Finish test load data")
    
    # # AD label 0, NC label 1
    # print("Start loading image")
    # TRAIN_PATH = "./AD_NC/train"


    
    # image = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transform)

    # print(type(image))

    

    # last_index = i - 1
    # print(image.imgs[last_index:(last_index + 2)])
    # print(image.imgs.index(('./AD_NC/train/AD/218391_78.jpeg', 0)))


    # mid_way = last_index // 2
    # print(mid_way)
    # current_patient = image.imgs[mid_way][0].split("/")[-1].split("_")[0]
    # previous_patient = image.imgs[mid_way - 1][0].split("/")[-1].split("_")[0]
    # print(current_patient)
    # print(previous_patient)
    # while current_patient == previous_patient:
    #     mid_way -= 1
    #     current_patient = image.imgs[mid_way][0].split("/")[-1].split("_")[0]
    #     previous_patient = image.imgs[mid_way - 1][0].split("/")[-1].split("_")[0]

    # print(mid_way)
    # # while image.imgs[mid_way][0]:
    # #     pass
    
    # # print(image.imgs[0])
    # # print(image.imgs])
    # # print(image[21519])
    # print("Finish")
    # transform = transforms.Compose([
    #         transforms.Resize(105),
    #         transforms.CenterCrop(105),
    #         transforms.ToTensor()
    #     ])
    # image = torchvision.datasets.ImageFolder(root=TRAIN_PATH, transform=transform)

    # current_label = 0
    # for i, (path, label) in enumerate(image.imgs):
    #     if i == 0:
    #         current_label = label
        
    #     if label != current_label:
    #         break
    # size_AD = i
    # size_NC = len(image) - size_AD
    # print(image.imgs[size_AD])
    # print(size_AD)
    # print(size_NC)
    # AD_patient = [[]]
    # NC_patient = []

    # # print(image[:size_AD])
    # range_list = list(range(0, size_AD))
    # trainset_1 = CustomSubset(image, range_list)

    # print(len(trainset_1))
    # print(trainset_1[-1])
    
    # print(type(image))
    # start_index = 0
    # current_patient = image.imgs[start_index][0].split("/")[-1].split("_")[0]

    # current_index = 0
    # for i in range(size_AD):
    #     patient_image = []
        
    #     for j in range(i, size_AD):
    #         pass


    # for i in range(len(size_AD)):
    #     if i == start_index: 
    #         continue
            
    #     patient = image.imgs[i][0].split("/")[-1].split("_")[0]
    #     if patient != current_patient:
    #         # range_list = list(range(start_index, i))
    #         # AD_patient.append(CustomSubset(image, range_list))
    #         pass
        
    # for img in image[:size_AD]:

    pass
