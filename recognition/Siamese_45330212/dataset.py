#Contains the data loader for loading and preprocessing your data
from torchvision import datasets
import torchvision.transforms as transforms
import random
from PIL import Image
from modules import *

class CustomTripletSiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Create a list of image paths
        self.image_paths = []
        self.labels = []
        folders = os.listdir(root_dir)
        print("> Creating image paths")
        for i, folder1 in enumerate(folders):
            folder2 = folders[i ^ 1]
            c = 0
            print("Folder:", folder1, folder2)

            folder1_path = os.path.join(root_dir, folder1)
            folder2_path = os.path.join(root_dir, folder2)

            folder1_images = os.listdir(folder1_path)
            folder2_images = os.listdir(folder2_path)

            for anchor in folder1_images:
                c += 1
                if c % 1000 == 0:
                    print("Count:", c)
                pos = random.choice(folder1_images)
                while anchor == pos:
                    print("FOUND SAME IMAGE - SHOULDN'T HAPPEN OFTEN")
                    pos = random.choice(folder1_images)
                neg = random.choice(folder2_images)

                anchor_path = os.path.join(folder1_path, anchor)
                pos_path = os.path.join(folder1_path, pos)
                neg_path = os.path.join(folder2_path, neg)

                self.image_paths.append((anchor_path, pos_path, neg_path, i))

        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
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
    def __init__(self, train_subset, model, device):
        self.model = model

        # Create a list of image paths
        self.image_embeddings = []
        print("> Creating image embedding")
        c = 0
        for anchor, pos, neg, label in train_subset:
            # print("Before shapes:", anchor.shape, pos.shape, neg.shape)
            anchor = anchor.unsqueeze(0)
            pos = torch.zeros_like(anchor)
            # neg = neg.unsqueeze(0)
            neg = torch.zeros_like(anchor)
            # print("After shapes:", anchor.shape, pos.shape, neg.shape, "\n")
            c += 1
            if c % 1000 == 0:
                print("Count:", c)
            output_anchor, _, _ = model(anchor.to(device), pos.to(device), neg.to(device))
            self.image_embeddings.append((output_anchor, label))

        print("< Finished creating image embeddings. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_embeddings)

    def __getitem__(self, index):
        return self.image_embeddings[index]

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.11550809, 0.11550809, 0.11550809), (0.22545652, 0.22545652, 0.22545652)),
])

print("> Getting triplet train set")
triplet_trainset = CustomTripletSiameseNetworkDataset(root_dir=Config.training_dir, transform=transform_train)
triplet_train_loader = torch.utils.data.DataLoader(triplet_trainset, batch_size=Config.train_batch_size, shuffle=True)
print("< Finished getting triplet train set")

print("> Getting train and validation set")
trainset = datasets.ImageFolder(root=Config.training_dir, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True)
# Calculate the lengths of the splits
total_len = len(trainset)
train_len = int(0.8 * total_len)
val_len = total_len - train_len

# Split the dataset
train_subset, val_subset = utils.random_split(trainset, [train_len, val_len])

# Create DataLoaders for the subsets
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=Config.train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=Config.train_batch_size, shuffle=False)
print("< Finished getting train set and validation set. With testing size:", len(train_subset), "and validation size:", len(val_subset))

print("> Getting test set")
# testset = CustomSiameseNetworkDataset(root_dir=Config.testing_dir, transform=transform_train)
testset = datasets.ImageFolder(Config.testing_dir, transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=Config.train_batch_size, shuffle=True)
print("< Finished getting test set")