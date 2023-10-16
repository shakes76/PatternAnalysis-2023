'''Data loader for loading and preprocessing data'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,ConcatDataset,Dataset,TensorDataset
from torchdata.datapipes.map import SequenceWrapper
import random

# Load and return normalized data
def normalise_data(path, size):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.Resize(size),  # img_size should be a tuple like (128, 128) actual img(256x240)
        transforms.ToTensor(),
        # You can add more transformations if needed
    ])

    
    # raw_dataset = datasets.ImageFolder(root=path)
    dataset = datasets.ImageFolder(root=path, transform=transform)
    


    # Return the data set of Img, Label
    return dataset

def load_data(dataset,batch_size, num_worker = 0, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=num_worker,shuffle=shuffle)
    return dataloader

# NOT in use
def filter_labels(dataset):
    images0,labels0 = [],[]
    images1,labels1 = [],[]
    for img,lbl in dataset:
        if lbl == 0:
            images0.append(img)
            labels0.append(lbl)
        else:
            images1.append(img)
            labels1.append(lbl)
    tensor_images0 = torch.stack(images0)
    tensor_labels0 = torch.tensor(labels0)

    tensor_images1 = torch.stack(images1)
    tensor_labels1 = torch.tensor(labels1)

    return TensorDataset(tensor_images0, tensor_labels0), TensorDataset(tensor_images1, tensor_labels1)

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


# def make_pair(dataset1, dataset2):

#     positive_pair1 = PairDataset(dataset1,dataset1,0)
#     positive_pair2 = PairDataset(dataset2,dataset2,0)
#     negative_pair1 = PairDataset(dataset1,dataset2,1)
#     negative_pair2 = PairDataset(dataset2,dataset1,1)
#     return positive_pair1,positive_pair2,negative_pair1,negative_pair2


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

# def visualise(img1, img2, labels, to_show=6, num_col=3, prediction=None, test=False):

#     num_row = to_show // num_col if to_show // num_col != 0 else 1

#     to_show = num_row * num_col

#     # Plot the images
#     fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
#     for i in range(to_show):

#         # If the number of rows is 1, the axes array is one-dimensional
#         if num_row == 1:
#             ax = axes[i % num_col]
#         else:
#             ax = axes[i // num_col, i % num_col]

#         ax.imshow((tf.concat([img1[i], img2[i]], axis=1).numpy()*255.0).astype("uint8"))
#         ax.set_axis_off()
#         if test:
#             ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
#         else:
#             ax.set_title("Label: {}".format(labels[i]))
#     if test:
#         plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
#     else:
#         plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

def visualise_1(dataset):
    img,lab = random.choice(dataset)
    plt.title(lab)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    
def visualise_batch(dataloader):
    LABELS = ['AD','ND']

    example_batch = iter(dataloader)
    images,labels = next(example_batch)

    plt.figure(figsize=(8,2)) # width x height
    batch_size = dataloader.batch_size
    for idx in range(batch_size):

        image = transforms.ToPILImage()(images[idx])
        label = LABELS[labels[idx].item()]

        plt.subplot(2,8,idx+1)
        
        plt.imshow(image)
        plt.title(label)
        plt.axis('off')



