from dataset import patient_split, SiameseDataSet, compose_transform
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

TEST_FILE_ROOT = "./AD_NC/test"
# 256x240

def load():
    train, val = patient_split()
    transform = compose_transform()
    TrainSet = SiameseDataSet(train, transform)
    ValidationSet = SiameseDataSet(val, transform)
    TestSet = SiameseDataSet(datasets.ImageFolder(root=TEST_FILE_ROOT), transform)



if __name__ == '__main__':
    pass
