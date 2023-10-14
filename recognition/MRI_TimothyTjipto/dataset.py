'''Data loader for loading and preprocessing data'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset
from torchdata.datapipes.map import SequenceWrapper

# Load and return normalized data
def normalise_data(path, size):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.Resize(size),  # img_size should be a tuple like (128, 128)
        transforms.ToTensor(),
        # You can add more transformations if needed
    ])

    

    dataset = datasets.ImageFolder(root=path, transform=transform)


    return dataset

# # DO NOT SHUFFLE
# def make_pair(dataset1, dataset2):
    
#     # postive_pair1 = torch.cat((dataset1,dataset1),1)
#     # postive_pair2 = torch.cat((dataset2,dataset2),1)
#     # negative_pair1 = torch.cat((dataset1,dataset2),1)
#     # negative_pair2 = torch.cat((dataset2,dataset1),1)

#     postive_pair1 = ConcatDataset([dataset1,dataset2])


class PairDataset(Dataset):
    def __init__(self,dataset1,dataset2, label):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.label = label

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))
    
    def __getitem__(self, index):
        img1,_ = self.dataset1[index]
        img2,_ = self.dataset2[index]

        return img1, img2, self.label


def make_pair(dataset1, dataset2):
    positive_pair1 = PairDataset(dataset1,dataset1,0)
    positive_pair2 = PairDataset(dataset2,dataset2,0)
    negative_pair1 = PairDataset(dataset1,dataset2,1)
    negative_pair2 = PairDataset(dataset2,dataset1,1)
    return positive_pair1,positive_pair2,negative_pair1,negative_pair2

def test_pair(test_dataset1, train_dataset1, test_dataset2, train_dataset2):
    test_positive_pair1 = PairDataset(test_dataset1,train_dataset1,0)
    test_positive_pair2 = PairDataset(test_dataset2,train_dataset2,0)
    test_negative_pair1 = PairDataset(test_dataset1,train_dataset2,1)
    test_negative_pair2 = PairDataset(test_dataset2,train_dataset1,1)
    return test_positive_pair1, test_positive_pair2, test_negative_pair1, test_negative_pair2


def shuffle(pos_pair1, pos_pair2, neg_pair1, neg_pair2):
    concatenated_dataset = ConcatDataset([pos_pair1, pos_pair2, neg_pair1, neg_pair2])
    return concatenated_dataset


def split_dataset(dataset, val_size, train_size):
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size,val_size])
    return  train_set,val_set 

def visualise(img1, img2, labels, to_show=6, num_col=3, prediction=None, test=False):

    num_row = to_show // num_col if to_show // num_col != 0 else 1

    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow((tf.concat([img1[i], img2[i]], axis=1).numpy()*255.0).astype("uint8"))
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

def load_data(dataset,)