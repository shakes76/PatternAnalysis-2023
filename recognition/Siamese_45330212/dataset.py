#Contains the data loader for loading and preprocessing your data
from torchvision import datasets
import torchvision.transforms as transforms
from modules import *

class CustomSiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Create a list of image paths
        self.image_paths = []
        self.labels = []
        folders = os.listdir(root_dir)
        print("> Creating image paths")
        for i, folder1 in enumerate(folders):
            for j, folder2 in enumerate(folders):
                c = 0
                print("Folder:", folder1, folder2)
                if i == j:
                    label = 0  # Images from the same folder
                else:
                    label = 1  # Images from different folders

                folder1_path = os.path.join(root_dir, folder1)
                folder2_path = os.path.join(root_dir, folder2)

                folder1_images = os.listdir(folder1_path)
                folder2_images = os.listdir(folder2_path)

                for img1 in folder1_images:
                    c += 1
                    if c % 1000 == 0:
                        print("Count:", c)
                    img2 = random.choice(folder2_images)
                    while img1 == img2 and i == j:
                        print("FOUND SAME IMAGE - SHOULDN'T HAPPEN OFTEN")
                        img2 = random.choice(folder2_images)

                    img1_path = os.path.join(folder1_path, img1)
                    img2_path = os.path.join(folder2_path, img2)

                    self.image_paths.append((img1_path, img2_path))
                    self.labels.append(label)

        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img1_path, img2_path = self.image_paths[index]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return img1, img2, label
    
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

                self.image_paths.append((anchor_path, pos_path, neg_path))

        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        anchor_path, pos_path, neg_path = self.image_paths[index]
        anchor = Image.open(anchor_path).convert("RGB")
        pos = Image.open(pos_path).convert("RGB")
        neg = Image.open(neg_path).convert("RGB")

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anchor, pos, neg

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.11550809, 0.11550809, 0.11550809), (0.22545652, 0.22545652, 0.22545652)),
])
print("> Getting train set")
trainset = CustomSiameseNetworkDataset(root_dir=Config.training_dir, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True)
print("< Finished getting train set\n")
print("> Getting triplet train set")
triplet_trainset = CustomTripletSiameseNetworkDataset(root_dir=Config.training_dir, transform=transform_train)
triplet_train_loader = torch.utils.data.DataLoader(triplet_trainset, batch_size=Config.train_batch_size, shuffle=True)
print("< Finished getting triplet train set\n")
print("> Getting test set")
# testset = CustomSiameseNetworkDataset(root_dir=Config.testing_dir, transform=transform_train)
testset = datasets.ImageFolder(Config.testing_dir, transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=Config.train_batch_size, shuffle=True)
print("< Finished getting test set")