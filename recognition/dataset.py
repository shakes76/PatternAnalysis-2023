from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx], self.label[idx]
