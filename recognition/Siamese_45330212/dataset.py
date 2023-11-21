#Contains the data loader for loading and preprocessing your data
from torchvision import datasets
import torchvision.transforms as transforms
import random
from PIL import Image
from modules import *

# Define the custom dataset for the Triplet Siamese Network
class CustomTripletSiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # Root directory for the dataset
        self.transform = transform# Transformations to apply on images

        # Initialize empty lists to store image paths and labels
        self.image_paths = []
        self.labels = []

        # Loop through each folder in the root directory
        folders = os.listdir(root_dir)
        print("\n> Creating image paths")
        for i, folder1 in enumerate(folders):
            folder2 = folders[i ^ 1]
            c = 0
            print("Folder:", folder1, folder2)

            folder1_path = os.path.join(root_dir, folder1)
            folder2_path = os.path.join(root_dir, folder2)

            folder1_images = os.listdir(folder1_path)
            folder2_images = os.listdir(folder2_path)

            # Create anchor-positive-negative triples from images
            for anchor in folder1_images:
                c += 1
                if c % 1000 == 0:
                    print("Count:", c)

                # Choose positive and negative examples
                pos = random.choice(folder1_images)
                while anchor == pos:
                    print("FOUND SAME IMAGE - SHOULDN'T HAPPEN OFTEN")
                    pos = random.choice(folder1_images)
                neg = random.choice(folder2_images)

                # Store the paths of anchor, positive and negative images
                anchor_path = os.path.join(folder1_path, anchor)
                pos_path = os.path.join(folder1_path, pos)
                neg_path = os.path.join(folder2_path, neg)

                self.image_paths.append((anchor_path, pos_path, neg_path, i))

        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths) # Return the total number of images

    def __getitem__(self, index):
        # Return a single item (anchor, positive, negative, label) at the given index
        anchor_path, pos_path, neg_path, label = self.image_paths[index]
        anchor = Image.open(anchor_path).convert("RGB")
        pos = Image.open(pos_path).convert("RGB")
        neg = Image.open(neg_path).convert("RGB")

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg, label
    
class CustomClassifcationDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=False, validateSet = set()):
        self.root_dir = root_dir # Root directory for the dataset
        self.transform = transform# Transformations to apply on images

        # Initialize empty lists to store image paths and labels
        self.image_paths = []
        self.labels = []
        self.totalPatients0 = set()
        self.trainPatients0 = set()
        self.valPatients0 = set()

        self.totalPatients1 = set()
        self.trainPatients1 = set()
        self.valPatients1 = set()

        # Loop through each folder in the root directory
        folders = os.listdir(root_dir)
        print("\n> Creating image paths", str(train))
        if train:
            for i, folder1 in enumerate(folders):
                folder1_path = os.path.join(root_dir, folder1)
                folder1_images = os.listdir(folder1_path)
                for image in folder1_images:
                    patient = image.split("_")[0]
                    if i == 0:
                        self.totalPatients0.add(patient)
                    else:
                        self.totalPatients1.add(patient)
            for i, p in enumerate(self.totalPatients0):
                if i <= int(len(self.totalPatients0) * 0.8):
                    self.trainPatients0.add(p)
                else:
                    self.valPatients0.add(p)
            for i, p in enumerate(self.totalPatients1):
                if i <= int(len(self.totalPatients1) * 0.8):
                    self.trainPatients1.add(p)
                else:
                    self.valPatients1.add(p)
            for i, folder1 in enumerate(folders):
                c = 0
                print("Folder:", folder1)
                folder1_path = os.path.join(root_dir, folder1)
                folder1_images = os.listdir(folder1_path)
                for image in folder1_images:
                    patient = image.split("_")[0]
                    if patient in self.trainPatients0 or patient in self.trainPatients1:
                        c += 1
                        if c % 1000 == 0:
                            print("Count:", c)
                        
                        image_path = os.path.join(folder1_path, image)
                        self.image_paths.append((image_path, i))
        else:
            for i, folder1 in enumerate(folders):
                c = 0
                print("Folder:", folder1)
                folder1_path = os.path.join(root_dir, folder1)
                folder1_images = os.listdir(folder1_path)
                for image in folder1_images:
                    patient = image.split("_")[0]
                    if patient in validateSet:
                        c += 1
                        if c % 1000 == 0:
                            print("Count:", c)
                        image_path = os.path.join(folder1_path, image)
                        self.image_paths.append((image_path, i))

        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths) # Return the total number of images

    def __getitem__(self, index):
        # Return a single item (anchor, positive, negative, label) at the given index
        image_path,  label = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def getValPatients(self):
        return (self.valPatients0.union(self.valPatients1))

# Initialize transform for training images
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.11550809, 0.11550809, 0.11550809), (0.22545652, 0.22545652, 0.22545652)),
])

transform_train_cropped = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.11550809, 0.11550809, 0.11550809), (0.22545652, 0.22545652, 0.22545652)),
    transforms.RandomCrop((240, 256), padding=10, fill=0),
])

print("> Getting triplet train set")
triplet_trainset0 = CustomTripletSiameseNetworkDataset(root_dir=Config.training_dir, transform=transform_train)
triplet_trainset_cropped = CustomTripletSiameseNetworkDataset(root_dir=Config.training_dir, transform=transform_train_cropped)
triplet_trainset = utils.ConcatDataset([triplet_trainset0, triplet_trainset_cropped])
# # Calculate the lengths of the splits for training and validation sets
# total_len = len(triplet_trainset)
# train_len = int(0.8 * total_len)
# val_len = total_len - train_len

# # Split the dataset into training and validation sets
# triplet_train_subset, triplet_val_subset = utils.random_split(triplet_trainset, [train_len, val_len])

# Create DataLoaders for the training and validation subsets
triplet_train_loader = torch.utils.data.DataLoader(triplet_trainset, batch_size=Config.siamese_train_batch_size, shuffle=True)
# triplet_val_loader = torch.utils.data.DataLoader(triplet_val_subset, batch_size=Config.siamese_train_batch_size, shuffle=False)
print("< Finished getting triplet train set")

print("> Getting train and validation set")
train_set_normal = CustomClassifcationDataset(root_dir=Config.training_dir, transform=transform_train, train=True)
train_set_cropped = CustomClassifcationDataset(root_dir=Config.training_dir, transform=transform_train_cropped, train=True)
trainset = utils.ConcatDataset([train_set_normal, train_set_cropped])

val_set_normal = CustomClassifcationDataset(root_dir=Config.training_dir, transform=transform_train, train=False, validateSet=train_set_normal.getValPatients())
val_set_cropped = CustomClassifcationDataset(root_dir=Config.training_dir, transform=transform_train_cropped, train=False, validateSet=train_set_normal.getValPatients())
valset = utils.ConcatDataset([val_set_normal, val_set_cropped])

# trainset0 = datasets.ImageFolder(root=Config.training_dir, transform=transform_train)
# cropped_trainset = datasets.ImageFolder(root=Config.training_dir, transform=transform_train_cropped)
# trainset = utils.ConcatDataset([trainset0, cropped_trainset])
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True)
# # Calculate the lengths of the splits
# total_len = len(trainset)
# train_len = int(0.8 * total_len)
# val_len = total_len - train_len

# # Split the dataset
# train_subset, val_subset = utils.random_split(trainset, [train_len, val_len])

# Create DataLoaders for the subsets
train_loader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=Config.train_batch_size, shuffle=False)
print("< Finished getting train set and validation set. With testing size:", len(trainset), "and validation size:", len(valset))

# Create DataLoader for the test set
print("> Getting test set")
# testset = CustomSiameseNetworkDataset(root_dir=Config.testing_dir, transform=transform_train)
testset = datasets.ImageFolder(Config.testing_dir, transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=Config.train_batch_size, shuffle=True)
print("< Finished getting test set")