''' This file contains all the misc functions and classes used to develop the model, every major function
is attached with a header comment denoting it's functionality'''

import os
import shutil

## The below commented code block calculates the confusion matrix and the necessary classification details 

# from modules import CCT
# from dataset import testloader
# from train import criterion
# import os
# import argparse
# import csv
# import time
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd

# import matplotlib.pyplot as plt
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt1
# device = torch.device("cuda")
# model = CCT(
#         img_size = (256, 256),
#         embedding_dim = 192,
#         n_conv_layers = 2,
#         kernel_size = 7,
#         stride = 2,
#         padding = 3,
#         pooling_kernel_size = 3,
#         pooling_stride = 2,
#         pooling_padding = 1,
#         num_layers = 2,
#         num_heads = 6,
#         mlp_ratio = 3.,
#         num_classes = 2,
#         positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
#     )
# model=(torch.load("/home/Student/s4737925/Project/PatternAnalysis-2023/recognition/ADNI_TRANSFORMER_47379251/model_90.pth"))
# model.to(device)
# model.eval()

# y_pred = []
# y_true = []
# classes = ('AD', 'NC')
# with torch.no_grad():
#         test_loss = 0
#         correct = 0
#         total = 0
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             output = model(inputs)
            
#             output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#             y_pred.extend(output) # Save Prediction
            
#             labels = targets.data.cpu().numpy()
#             y_true.extend(labels) # Save Truth
 
# cf_matrix = confusion_matrix(y_true, y_pred)
# CM = cf_matrix
# df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
#                      columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('confusion.png')

# tn=CM[0][0]
# tp=CM[1][1]
# fp=CM[0][1]
# fn=CM[1][0]
# acc=np.sum(np.diag(CM)/np.sum(CM))
# sensitivity=tp/(tp+fn)
# precision=tp/(tp+fp)
        
# print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
# print('Confusion Matirx : ')
# print(CM)
# print('- Sensitivity : ',(tp/(tp+fn))*100)
# print('- Specificity : ',(tn/(tn+fp))*100)
# print('- Precision: ',(tp/(tp+fp))*100)
# print('- NPV: ',(tn/(tn+fn))*100)
# print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
## DATA SPLIT
# source_directory = '/home/Student/s4737925/Project/Main/AD'
# destination_directory = '/home/Student/s4737925/Project/Main/AD_NEW'
# for filename in os.listdir(source_directory):
#     if os.path.isfile(os.path.join(source_directory, filename)):
#         prefix = filename.split('_')[0]  # Extract the "xxx" part from the filename

#         # Create a subdirectory for the prefix if it doesn't exist
#         subdirectory_path = os.path.join(destination_directory, prefix)
#         os.makedirs(subdirectory_path, exist_ok=True)

#         # Move the file to the corresponding subdirectory
#         file_path = os.path.join(source_directory, filename)
#         destination_path = os.path.join(subdirectory_path, filename)

#         if not os.path.exists(destination_path):
#             shutil.move(file_path, destination_path)
# print("DONE")

# import os
# import shutil
# import random

# # Set the source directory and create destination directories
# source_dir = '/home/Student/s4737925/Project/Main/NC_NEW'
# train_dir = '/home/Student/s4737925/Project/Patient_Split/train/NC'
# test_dir = '/home/Student/s4737925/Project/Patient_Split/test/NC'
# valid_dir = '/home/Student/s4737925/Project/Patient_Split/valid/NC'

# # Set the split percentages
# train_percentage = 0.7
# test_percentage = 0.15
# valid_percentage = 0.15

# # Create the destination directories
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
# os.makedirs(valid_dir, exist_ok=True)

# # Get a list of all folders in the source directory
# folders = os.listdir(source_dir)

# # Shuffle the folders randomly
# random.shuffle(folders)

# # Calculate the split points
# total_folders = len(folders)
# train_split = int(total_folders * train_percentage)
# test_split = int(total_folders * (test_percentage))

# # Split the folders into train, test, and valid sets
# train_folders = folders[:train_split]
# test_folders = folders[train_split:test_split]
# valid_folders = folders[test_split:]

# print()

# # Move the folders to their respective directories
# for folder in train_folders:
#     source_path = os.path.join(source_dir, folder)
#     dest_path = os.path.join(train_dir, folder)
#     shutil.move(source_path, dest_path)

# for folder in test_folders:
#     source_path = os.path.join(source_dir, folder)
#     dest_path = os.path.join(test_dir, folder)
#     shutil.move(source_path, dest_path)

# for folder in valid_folders:
#     source_path = os.path.join(source_dir, folder)
#     dest_path = os.path.join(valid_dir, folder)
#     shutil.move(source_path, dest_path)

# print("Splitting complete.")

# import os
# import shutil
# import random

# source_folders = ['AD', 'NC']
# destination_folder = 'Split_Data_Main'

# # Create the destination folder if it doesn't exist
# os.makedirs(destination_folder, exist_ok=True)

# # Define the train, test, and valid ratios (70%, 15%, and 15% respectively):
# train_ratio = 0.7
# test_ratio = 0.15
# valid_ratio = 0.15

# # Function to split data by ID and further split by train, test, and valid
# def split_data_by_id(source_folder, id_folder):
#     files = os.listdir(source_folder)
#     num_files = len(files)
    
#     # Calculate the number of files for each split
#     num_train = int(num_files * train_ratio)
#     num_test = int(num_files * test_ratio)
#     num_valid = num_files - num_train - num_test
    
#     # Shuffle the files for randomness
#     random.shuffle(files)
    
#     # Split the files into train, test, and valid sets
#     train_set = files[:num_train]
#     test_set = files[num_train:num_train + num_test]
#     valid_set = files[num_train + num_test:]
    
#     # Create subfolders for train, test, and valid within the ID folder
#     os.makedirs(os.path.join(id_folder, 'train'), exist_ok=True)
#     os.makedirs(os.path.join(id_folder, 'test'), exist_ok=True)
#     os.makedirs(os.path.join(id_folder, 'valid'), exist_ok=True)
    
#     # Move the files to their respective folders
#     for file in train_set:
#         source_file = os.path.join(source_folder, file)
#         destination_file = os.path.join(id_folder, 'train', file)
#         shutil.copy(source_file, destination_file)
#     for file in test_set:
#         source_file = os.path.join(source_folder, file)
#         destination_file = os.path.join(id_folder, 'test', file)
#         shutil.copy(source_file, destination_file)
#     for file in valid_set:
#         source_file = os.path.join(source_folder, file)
#         destination_file = os.path.join(id_folder, 'valid', file)
#         shutil.copy(source_file, destination_file)

# # Split data for all source folders
# for source_folder in source_folders:
#     files = os.listdir(source_folder)
    
#     for file in files:
#         id = file.split('_')[0]  # Extract the ID before the underscore
#         id_folder = os.path.join(destination_folder, id)
        
#         # Create a folder for the ID if it doesn't exist
#         os.makedirs(id_folder, exist_ok=True)
        
#         source_file = os.path.join(source_folder, file)
#         destination_file = os.path.join(id_folder, file)
        
#         # Move the file to the folder corresponding to its ID
#         shutil.move(source_file, destination_file)
        
#     # Split the data within the ID folder
#     split_data_by_id(id_folder, id_folder)

# print("Data splitting and organization complete.")

# import os
# import shutil
# import random

# # Define the source directory containing the 1526 directories
# source_directory = "/home/Student/s4737925/Project/Main/NC_NEW"

# # Define the destination directories
# train_directory = "/home/Student/s4737925/Project/Main/train/NC"
# test_directory = "/home/Student/s4737925/Project/Main/test/NC"
# valid_directory = "/home/Student/s4737925/Project/Main/valid/NC"

# # Define the split percentages
# train_percent = 0.7
# test_percent = 0.15
# valid_percent = 0.15

# # Create destination directories if they don't exist
# os.makedirs(train_directory, exist_ok=True)
# os.makedirs(test_directory, exist_ok=True)
# os.makedirs(valid_directory, exist_ok=True)

# # List all directories in the source directory
# directories = os.listdir(source_directory)

# # Shuffle the list of directories randomly
# random.shuffle(directories)

# # Calculate the split points
# train_end = int(train_percent * len(directories))
# test_end = train_end + int(test_percent * len(directories))

# # Split directories and move them to the appropriate destination
# for i, directory in enumerate(directories):
#     source_path = os.path.join(source_directory, directory)
#     if i < train_end:
#         destination_path = os.path.join(train_directory, directory)
#     elif i < test_end:
#         destination_path = os.path.join(test_directory, directory)
#     else:
#         destination_path = os.path.join(valid_directory, directory)
    
#     shutil.move(source_path, destination_path)
# print("DONE")


## AUGMENTATION (CUSTOM)
import random
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

from modules import *
from dataset import *


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''



## NORMALIZATION
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# ## Plotting Graphs
# import matplotlib.pyplot as plt
# a = [62.89473684210526, 72.25, 71.97435897435898, 72.56410256410257, 71.0, 74.1, 74.0, 74.32352941176471, 73.34285714285714, 74.43333333333334, 74.75, 76.05263157894737, 74.21052631578948, 74.0, 74.8, 74.47619047619048, 75.5, 72.1590909090909, 74.0, 74.54545454545455, 75.52631578947368, 73.57894736842105, 76.66666666666667, 76.94736842105263, 76.47368421052632, 77.6842105263158, 77.25, 75.0, 75.78947368421052, 76.66666666666667, 73.33333333333333, 74.95652173913044, 77.25, 75.25, 76.42105263157895, 76.62857142857143, 74.66666666666667, 75.57894736842105, 76.66666666666667, 75.66666666666667, 77.33333333333333, 74.05263157894737, 77.0, 76.85714285714286, 76.27586206896552, 76.66666666666667, 76.33333333333333, 77.66666666666667, 76.66666666666667, 76.33333333333333, 77.0, 77.33333333333333, 76.66666666666667, 76.0, 77.33333333333333, 76.33333333333333, 78.66666666666667, 78.0, 76.66666666666667, 77.66666666666667, 79.33333333333333, 75.66666666666667, 76.0, 77.33333333333333, 77.0, 76.0, 76.33333333333333, 76.0, 77.0, 79.33333333333333, 77.33333333333333, 78.33333333333333, 78.66666666666667, 77.33333333333333, 78.66666666666667, 77.66666666666667, 77.66666666666667, 78.33333333333333, 78.33333333333333, 78.0, 78.0, 76.66666666666667, 77.33333333333333, 78.0, 76.66666666666667, 79.0, 78.66666666666667, 78.0, 77.33333333333333, 78.0, 77.0, 78.0, 78.33333333333333, 77.66666666666667, 78.66666666666667, 77.66666666666667, 77.33333333333333, 77.66666666666667, 77.66666666666667, 77.66666666666667]
# print(a.index(max(a)), max(a))
# a1=[62.89473684210526, 70.13636363636364, 71.57777777777778, 71.95454545454545, 70.26315789473684, 73.65217391304348, 74.33333333333333, 75.2051282051282, 75.125, 73.125, 74.875, 77.0, 76.33333333333333, 73.84210526315789, 76.21052631578948, 74.4, 76.8, 74.33333333333333, 73.26666666666667, 73.33333333333333, 75.10526315789474, 77.0, 74.88888888888889, 77.26315789473684, 75.78260869565217, 77.78947368421052, 77.66666666666667, 74.73684210526316, 77.66666666666667, 77.0, 74.0, 74.66666666666667, 76.2, 76.73684210526316, 75.94736842105263, 77.8, 74.33333333333333, 77.33333333333333, 72.66666666666667, 78.0, 77.0, 77.0, 78.625, 77.15789473684211, 77.42105263157895, 76.21052631578948, 76.15789473684211, 78.3157894736842, 75.94736842105263, 76.66666666666667, 76.84210526315789, 75.84210526315789, 76.42105263157895, 77.15789473684211, 76.66666666666667, 75.36842105263158, 77.89473684210526, 75.0, 75.33333333333333, 78.0, 76.94736842105263, 78.0, 76.66666666666667, 75.18181818181819, 76.89473684210526, 76.84210526315789, 77.0, 76.33333333333333, 76.48387096774194, 77.33333333333333, 76.90322580645162, 77.0, 75.17142857142858, 76.10526315789474, 76.0, 76.73684210526316, 76.78947368421052, 77.15789473684211, 76.6842105263158, 77.0, 76.56521739130434, 77.57894736842105, 76.36842105263158, 77.89473684210526, 76.33333333333333, 77.05263157894737, 76.0, 77.10526315789474, 74.33333333333333, 77.0, 76.0, 75.77142857142857, 76.33333333333333, 76.0, 76.6842105263158, 77.66666666666667, 77.33333333333333, 75.57894736842105, 76.33333333333333, 77.36842105263158, 77.18181818181819, 77.52631578947368, 76.0, 77.0, 76.89473684210526, 78.10526315789474, 77.78947368421052, 77.36842105263158, 76.89473684210526, 76.94736842105263, 76.05263157894737, 77.0, 75.33333333333333, 77.05263157894737, 76.57894736842105, 77.0, 78.15789473684211, 77.13636363636364, 76.73684210526316, 76.26315789473684, 76.84210526315789, 76.47368421052632, 77.84210526315789, 77.26315789473684, 76.73684210526316, 76.0, 76.66666666666667, 76.52631578947368, 77.52631578947368, 76.47368421052632, 77.31818181818181, 76.94736842105263, 77.3157894736842, 76.6842105263158, 76.0, 77.33333333333333, 77.0, 77.33333333333333, 77.33333333333333, 76.89473684210526, 77.33333333333333, 77.0, 77.10526315789474, 77.0, 77.0, 77.0, 77.15789473684211, 77.0, 77.0, 77.0]
# print(a1.index(max(a1)), max(a1))
# a2=[72.0, 70.71428571428571, 72.3, 69.8974358974359, 71.0, 75.0, 77.0, 73.78947368421052, 77.0, 79.0, 74.42105263157895, 74.66666666666667, 70.26315789473684, 74.0, 74.42105263157895, 74.63157894736842, 74.33333333333333, 75.47368421052632, 77.26315789473684, 76.33333333333333, 76.66666666666667, 76.0, 74.8, 75.3157894736842, 77.66666666666667, 77.0, 75.66666666666667, 78.0, 75.56521739130434, 75.6842105263158, 75.10526315789474, 76.0, 76.0, 76.0, 78.33333333333333, 74.54285714285714, 76.66666666666667, 75.33333333333333, 74.62857142857143, 75.0, 77.66666666666667, 76.66666666666667, 75.33333333333333, 77.66666666666667, 76.66666666666667, 75.05714285714286, 76.84210526315789, 76.66666666666667, 78.66666666666667, 78.0, 76.66666666666667, 76.0, 74.57142857142857, 76.0, 76.66666666666667, 76.33333333333333, 76.66666666666667, 76.66666666666667, 76.66666666666667, 76.33333333333333, 76.66666666666667, 76.21052631578948, 75.66666666666667, 76.73684210526316, 75.0, 76.66666666666667, 77.66666666666667, 75.33333333333333, 77.0, 77.0, 76.0, 76.0, 76.0, 76.33333333333333, 76.33333333333333, 76.66666666666667, 76.0, 75.88571428571429, 76.0, 76.0]
# print(a2.index(max(a2)), max(a2))
# a3=[52.473684210526315, 73.0, 73.19444444444444, 72.4, 69.04545454545455, 73.43589743589743, 72.75, 74.8, 73.2, 72.7948717948718, 74.57894736842105, 74.94117647058823, 75.66666666666667, 71.0, 74.48387096774194, 74.35483870967742, 75.05263157894737, 70.6, 75.0, 75.52631578947368, 72.5609756097561, 76.47368421052632, 75.57894736842105, 77.33333333333333, 75.66666666666667, 76.36666666666666, 75.89473684210526, 75.0, 76.73684210526316, 75.0, 75.6774193548387, 75.6, 74.85714285714286, 75.47368421052632, 76.04878048780488, 75.0, 74.66666666666667, 76.3157894736842, 76.42105263157895, 76.28571428571429, 76.6842105263158, 77.05263157894737, 77.03225806451613, 76.36842105263158, 77.03225806451613, 75.66666666666667, 76.89473684210526, 77.88571428571429, 75.89473684210526, 77.14285714285714, 76.33333333333333, 76.66666666666667, 76.89473684210526, 78.66666666666667, 76.66666666666667, 76.2, 75.48571428571428, 76.22857142857143, 76.6, 74.6, 75.68571428571428, 76.93333333333334, 76.2, 77.28571428571429, 76.53333333333333, 75.66666666666667, 77.25806451612904, 77.4, 75.91428571428571, 77.57142857142857, 77.28571428571429, 77.66666666666667, 75.91111111111111, 77.22857142857143, 76.22857142857143, 77.02857142857142, 76.66666666666667, 78.0, 76.66666666666667, 77.66666666666667, 75.5, 75.0, 76.02857142857142, 76.0, 77.0, 76.28571428571429, 76.62857142857143, 75.51428571428572, 77.33333333333333, 76.3, 75.02439024390245, 77.74285714285715, 77.04878048780488, 77.51428571428572, 77.05263157894737, 74.86363636363636, 75.8, 76.0, 76.33333333333333, 76.66666666666667, 76.9090909090909, 75.37142857142857, 76.33333333333333, 76.6, 76.17142857142858, 77.33333333333333, 76.66666666666667, 75.42222222222222, 76.66666666666667, 77.66666666666667, 76.0, 75.66666666666667, 77.33333333333333, 75.58536585365853, 77.33333333333333, 76.11363636363636, 77.0, 77.6774193548387, 77.33333333333333, 76.66666666666667, 77.33333333333333, 76.66666666666667, 76.66666666666667, 76.33333333333333, 77.33333333333333, 76.66666666666667, 76.66666666666667, 76.73333333333333, 76.24444444444444, 76.57142857142857, 76.2, 77.0, 77.33333333333333, 76.14285714285714, 76.54285714285714, 77.11428571428571, 76.68571428571428, 76.6, 76.72727272727273, 76.66666666666667, 76.33333333333333, 77.0, 77.0, 76.4090909090909, 76.66666666666667, 76.66666666666667, 77.0, 76.66666666666667, 76.66666666666667, 76.66666666666667]
# print(a3.index(max(a3)), max(a3))
# a4=[50.57142857142857, 71.0, 71.03333333333333, 67.90243902439025, 69.63414634146342, 72.47368421052632, 71.26315789473684, 73.47826086956522, 71.42105263157895, 73.52173913043478, 73.0, 74.05263157894737, 74.1, 74.52631578947368, 72.42105263157895, 74.47368421052632, 72.0, 75.21052631578948, 74.57894736842105, 74.21052631578948, 76.10526315789474, 76.94736842105263, 73.0, 76.8, 75.66666666666667, 74.73684210526316, 77.33333333333333, 76.66666666666667, 77.0, 77.33333333333333, 75.0, 77.66666666666667, 76.66666666666667, 76.33333333333333, 78.33333333333333, 77.33333333333333, 76.33333333333333, 74.89473684210526, 79.66666666666667, 76.21052631578948, 79.0, 75.33333333333333, 79.66666666666667, 76.66666666666667, 77.0, 77.33333333333333, 74.33333333333333, 77.33333333333333, 77.52631578947368, 73.33333333333333, 80.66666666666667, 78.33333333333333, 76.66666666666667, 76.66666666666667, 76.10526315789474, 79.36842105263158, 79.0, 78.66666666666667, 76.15789473684211, 76.84210526315789, 78.33333333333333, 78.66666666666667, 78.0, 76.77142857142857, 79.0, 79.33333333333333, 78.33333333333333, 77.73684210526316, 77.0, 76.73684210526316, 77.57894736842105, 75.71428571428571, 77.0, 76.66666666666667, 76.6842105263158, 77.36842105263158, 78.05714285714286, 77.66666666666667, 77.66666666666667, 77.0, 77.33333333333333, 77.05263157894737, 77.10526315789474, 79.0, 77.3157894736842, 77.68571428571428, 77.26315789473684, 77.36842105263158, 77.17142857142858, 77.94736842105263, 78.15789473684211, 77.73684210526316, 78.36842105263158, 77.47368421052632, 77.82857142857142, 77.73684210526316, 77.26315789473684, 77.37142857142857, 77.63157894736842, 77.63157894736842]
# print(a4.index(max(a4)), max(a4))
# a5=[57.0, 72.55555555555556, 72.71111111111111, 71.15789473684211, 69.6, 73.425, 72.8, 75.28888888888889, 73.03030303030303, 74.35897435897436, 74.65217391304348, 74.65714285714286, 74.43478260869566, 71.02439024390245, 74.26315789473684, 74.36842105263158, 75.66666666666667, 72.04878048780488, 73.66666666666667, 75.2, 76.0, 76.6, 74.52173913043478, 77.66666666666667, 74.66666666666667, 74.84210526315789, 75.66666666666667, 73.84210526315789, 75.3157894736842, 75.22727272727273, 71.51219512195122, 74.0, 75.45454545454545, 76.47368421052632, 75.33333333333333, 74.33333333333333, 74.66666666666667, 78.0, 75.0, 74.66666666666667, 77.0, 77.66666666666667, 76.66666666666667, 77.66666666666667, 76.66666666666667, 77.0, 77.0, 76.33333333333333, 77.0, 76.82608695652173, 74.8695652173913, 75.0, 78.33333333333333, 77.66666666666667, 76.66666666666667, 76.69565217391305, 75.0, 74.66666666666667, 76.52631578947368, 75.52173913043478, 75.3170731707317, 76.51219512195122, 78.0, 77.0, 77.15789473684211, 76.77272727272727, 76.0909090909091, 77.0, 77.0, 76.84210526315789, 75.66666666666667, 76.66666666666667, 76.21052631578948, 75.63636363636364, 75.33333333333333, 76.26315789473684, 76.33333333333333, 76.36842105263158, 76.0, 76.07317073170732, 76.02439024390245, 76.33333333333333, 76.73684210526316, 76.15789473684211, 77.33333333333333, 76.05263157894737, 76.33333333333333, 77.0, 76.02439024390245, 76.6842105263158, 76.54545454545455, 76.5, 76.5, 76.66666666666667, 76.5, 76.1951219512195, 76.07317073170732, 76.66666666666667, 76.66666666666667, 76.66666666666667]
# print(a5.index(max(a5)), max(a5))
# a6=[51.9, 54.578947368421055, 69.82222222222222, 71.66666666666667, 72.15151515151516, 73.39393939393939, 72.8, 71.8, 71.63157894736842, 70.74193548387096, 72.0, 74.2, 72.66666666666667, 73.8, 73.33333333333333, 74.16666666666667, 74.33333333333333, 73.66666666666667, 75.0, 73.33333333333333, 75.0, 75.66666666666667, 75.66666666666667, 75.66666666666667, 74.57894736842105, 74.0, 75.66666666666667, 71.38636363636364, 75.21052631578948, 76.57894736842105, 75.6842105263158, 76.42105263157895, 74.0, 74.36842105263158, 76.42105263157895, 74.0, 74.1219512195122, 73.33333333333333, 77.15789473684211, 76.52631578947368, 75.0, 76.94736842105263, 76.66666666666667, 76.57777777777778, 76.57894736842105, 76.52631578947368, 77.21052631578948, 76.38709677419355, 73.26315789473684, 75.33333333333333, 74.31818181818181, 74.66666666666667, 76.26315789473684, 76.42105263157895, 75.33333333333333, 76.03333333333333, 74.4090909090909, 74.56818181818181, 77.33333333333333, 76.0, 75.45161290322581, 76.05263157894737, 75.625, 75.4, 76.66666666666667, 74.89473684210526, 77.21052631578948, 77.3157894736842, 75.05263157894737, 76.3, 76.02222222222223, 75.28571428571429, 76.43333333333334, 77.0, 76.33333333333333, 75.52631578947368, 76.66666666666667, 76.17142857142858, 76.35483870967742, 76.0, 75.48571428571428, 77.06451612903226, 76.66666666666667, 77.0, 76.66666666666667, 76.26666666666667, 75.48571428571428, 76.66666666666667, 76.0, 76.89473684210526, 76.33333333333333, 76.33333333333333, 75.90322580645162, 76.33333333333333, 76.33333333333333, 76.10526315789474, 76.33333333333333, 76.33333333333333, 76.33333333333333, 76.66666666666667]
# print(a6.index(max(a6)), max(a6))
# a7=[50.57142857142857, 63.333333333333336, 70.57575757575758, 68.21052631578948, 69.78260869565217, 68.51219512195122, 71.57894736842105, 72.25, 72.11111111111111, 73.10526315789474, 72.77777777777777, 73.0, 75.2, 74.0, 72.89473684210526, 74.13636363636364, 73.33333333333333, 74.42105263157895, 74.05263157894737, 75.36842105263158, 74.36842105263158, 74.15789473684211, 72.66666666666667, 75.33333333333333, 76.66666666666667, 76.58536585365853, 76.33333333333333, 74.66666666666667, 77.33333333333333, 76.66666666666667, 75.66666666666667, 76.33333333333333, 73.54285714285714, 75.66666666666667, 76.66666666666667, 77.66666666666667, 74.73684210526316, 75.33333333333333, 77.0, 73.66666666666667, 79.0, 73.0, 75.0, 77.0, 75.0, 76.66666666666667, 73.36363636363636, 78.33333333333333, 76.66666666666667, 73.63157894736842, 75.0, 77.0, 77.0, 77.33333333333333, 76.33333333333333, 76.33333333333333, 77.66666666666667, 77.66666666666667, 75.66666666666667, 77.0, 78.66666666666667, 78.33333333333333, 78.0, 77.0, 76.33333333333333, 78.33333333333333, 77.33333333333333, 77.33333333333333, 74.77777777777777, 75.66666666666667, 75.33333333333333, 74.5111111111111, 76.33333333333333, 76.66666666666667, 76.0, 76.37142857142857, 75.66666666666667, 76.6, 76.33333333333333, 76.35555555555555, 75.54285714285714, 76.0, 76.0, 76.66666666666667, 75.48571428571428, 75.88571428571429, 75.88571428571429, 75.6, 75.65714285714286, 76.66666666666667, 76.33333333333333, 75.71428571428571, 76.33333333333333, 75.60975609756098, 76.66666666666667, 75.97777777777777, 75.82857142857142, 76.33333333333333, 76.06666666666666, 76.0]
# print(a7.index(max(a7)), max(a7))
# a8=[50.57142857142857, 70.66666666666667, 73.25, 67.8, 72.0, 71.66666666666667, 71.63157894736842, 74.06451612903226, 74.33333333333333, 75.0, 74.14285714285714, 74.23333333333333, 76.0, 75.05263157894737, 71.26315789473684, 74.21739130434783, 76.0, 75.33333333333333, 76.66666666666667, 74.84210526315789, 79.66666666666667, 76.33333333333333, 76.33333333333333, 78.66666666666667, 76.5, 76.75, 78.66666666666667, 77.66666666666667, 79.33333333333333, 78.66666666666667, 74.33333333333333, 76.0, 80.33333333333333, 80.33333333333333, 78.66666666666667, 78.0, 77.66666666666667, 77.0, 79.0, 75.0, 78.0, 73.66666666666667, 76.66666666666667, 77.33333333333333, 78.66666666666667, 78.0, 77.0, 75.66666666666667, 79.0, 77.66666666666667, 79.66666666666667, 78.0, 78.66666666666667, 77.66666666666667, 75.33333333333333, 80.66666666666667, 79.33333333333333, 79.0, 74.70454545454545, 79.33333333333333, 80.0, 80.0, 77.33333333333333, 79.33333333333333, 77.66666666666667, 79.33333333333333, 79.33333333333333, 80.33333333333333, 78.33333333333333, 76.33333333333333, 77.0, 76.0, 79.33333333333333, 77.33333333333333, 78.0, 79.0, 78.0, 79.33333333333333, 79.33333333333333, 79.66666666666667, 77.66666666666667, 79.0, 79.33333333333333, 78.66666666666667, 78.0, 78.66666666666667, 78.66666666666667, 78.33333333333333, 76.66666666666667, 77.66666666666667, 78.33333333333333, 76.66666666666667, 78.33333333333333, 79.0, 78.0, 77.66666666666667, 77.0, 77.66666666666667, 77.66666666666667, 77.66666666666667]
# print(a8.index(max(a8)), max(a8))
# a9=[50.57142857142857, 68.79545454545455, 72.5, 68.54285714285714, 68.7, 70.15, 73.2, 73.0, 72.23809523809524, 74.2, 74.66666666666667, 74.15789473684211, 73.66666666666667, 75.8, 74.66666666666667, 74.55555555555556, 72.33333333333333, 75.10526315789474, 74.66666666666667, 74.66666666666667, 77.33333333333333, 75.33333333333333, 74.0, 75.0, 76.66666666666667, 75.36842105263158, 76.0, 76.66666666666667, 77.66666666666667, 76.33333333333333, 74.66666666666667, 77.0, 77.33333333333333, 75.33333333333333, 76.66666666666667, 77.33333333333333, 75.66666666666667, 75.0, 76.33333333333333, 77.0, 77.33333333333333, 74.66666666666667, 75.66666666666667, 78.0, 76.66666666666667, 77.66666666666667, 77.0, 76.33333333333333, 76.33333333333333, 76.0, 78.0, 76.33333333333333, 76.33333333333333, 77.66666666666667, 77.0]
# print(a9.index(max(a9)), max(a9))
# a10=[56.0, 57.0, 51.63157894736842, 68.64444444444445, 70.66666666666667, 71.22222222222223, 70.04545454545455, 70.0, 71.51282051282051, 71.13333333333334, 71.25, 71.75, 70.1590909090909, 74.66666666666667, 70.46341463414635, 74.62222222222222, 72.775, 73.26315789473684, 77.66666666666667, 75.66666666666667, 72.84210526315789, 73.0, 76.66666666666667, 72.2, 72.85, 75.84210526315789, 74.47368421052632, 74.34782608695652, 75.48571428571428, 77.78947368421052, 78.3, 76.73684210526316, 77.33333333333333, 79.63157894736842, 78.42857142857143, 78.26315789473684, 77.66666666666667, 77.10526315789474, 77.94736842105263, 78.95, 78.42105263157895, 72.10526315789474, 76.36842105263158, 78.66666666666667, 72.91304347826087, 80.0, 77.68181818181819, 81.0, 79.375, 73.89473684210526, 79.66666666666667, 80.66666666666667, 80.33333333333333, 79.66666666666667, 79.0, 79.0, 78.66666666666667, 78.875, 79.33333333333333, 80.6, 80.8, 79.66666666666667, 79.42105263157895, 80.0, 81.0, 80.66666666666667, 81.0, 79.33333333333333, 79.875, 78.125, 77.88888888888889, 80.33333333333333, 77.55555555555556, 77.36842105263158, 77.66666666666667, 79.125, 79.5, 79.89473684210526, 78.73684210526316, 77.89473684210526, 78.0, 78.5, 78.125, 78.42105263157895, 78.84210526315789, 78.0, 80.0, 80.0, 78.625, 78.125, 78.375, 78.875, 78.75, 78.375, 78.875, 78.875, 78.5, 78.875, 78.875, 78.75]
# print(a10.index(max(a10)), max(a10))
# a12=[56.0, 52.473684210526315, 64.25, 68.52631578947368, 68.66666666666667, 69.28571428571429, 72.0, 73.25, 73.0, 64.65, 64.2, 65.675, 74.75, 73.28205128205128, 71.66666666666667, 71.42105263157895, 72.36842105263158, 71.5, 75.33333333333333, 74.15789473684211, 74.63157894736842, 76.2, 73.36842105263158, 74.6842105263158, 74.6842105263158, 69.63414634146342, 71.69565217391305, 76.3157894736842, 74.84210526315789, 73.42857142857143, 73.63157894736842, 75.66666666666667, 75.66666666666667, 76.84210526315789, 74.45, 75.21052631578948, 73.0, 76.4, 73.57894736842105, 73.08695652173913, 74.08695652173913, 75.57894736842105, 77.0, 77.65714285714286, 74.31428571428572, 76.66666666666667, 75.0, 73.73333333333333, 74.33333333333333, 76.02857142857142, 78.33333333333333, 74.73684210526316, 78.66666666666667, 77.05, 74.88571428571429, 75.66666666666667, 77.33333333333333, 73.26315789473684, 73.0909090909091, 75.97142857142858, 75.69565217391305, 75.33333333333333, 78.0, 77.66666666666667, 78.66666666666667, 76.34782608695652, 77.33333333333333, 78.68571428571428, 77.0, 75.66666666666667, 76.33333333333333, 77.66666666666667, 78.33333333333333, 75.4090909090909, 75.9090909090909, 76.57142857142857, 76.48571428571428, 76.57894736842105, 75.88571428571429, 77.57894736842105, 76.28571428571429, 77.26315789473684, 77.69565217391305, 77.28571428571429, 77.52173913043478, 76.17142857142858, 77.42857142857143, 77.22857142857143, 76.69565217391305, 76.4, 76.97142857142858, 76.82857142857142, 76.42857142857143, 76.65714285714286, 76.42857142857143, 76.51428571428572, 76.6, 76.54285714285714, 76.45714285714286, 76.45714285714286]
# print(a12.index(max(a12)), max(a12))
# a11=[57.083333333333336, 59.0, 52.05263157894737, 63.0, 68.0, 69.48780487804878, 69.79545454545455, 71.0, 72.33333333333333, 69.88636363636364, 72.5, 72.37777777777778, 73.33333333333333, 71.925, 75.0, 73.2, 69.43181818181819, 73.3157894736842, 74.5, 73.33333333333333, 73.66666666666667, 75.05263157894737, 74.26666666666667, 73.4090909090909, 74.88636363636364, 75.36842105263158, 74.94736842105263, 73.2, 71.86666666666666, 73.81818181818181, 73.29545454545455, 74.94736842105263, 74.37142857142857, 72.25, 74.51219512195122, 74.48571428571428, 74.02857142857142, 73.0, 75.11111111111111, 76.0, 73.55555555555556, 76.08888888888889, 74.75, 73.0, 74.6, 75.66666666666667, 70.73170731707317, 75.61363636363636, 76.55, 76.33333333333333, 73.13333333333334, 76.56521739130434, 75.66666666666667, 76.6842105263158, 76.66666666666667, 74.82857142857142, 77.33333333333333, 77.21739130434783, 74.86363636363636, 76.11111111111111, 77.02222222222223, 76.48571428571428, 77.30434782608695, 74.73333333333333, 75.04545454545455, 76.14285714285714, 76.55555555555556, 75.77777777777777, 75.97777777777777, 76.08571428571429, 75.84444444444445, 74.73333333333333, 76.71111111111111, 76.66666666666667, 76.28888888888889, 76.46666666666667, 75.0909090909091, 76.2, 76.48571428571428, 75.17142857142858, 76.4, 76.4, 76.05714285714286, 76.25714285714285, 75.11428571428571, 76.57142857142857, 77.0, 76.06666666666666, 76.06666666666666, 76.33333333333333, 75.77142857142857, 76.31428571428572, 76.66666666666667, 76.2, 76.22857142857143, 76.66666666666667, 76.2, 76.17142857142858, 76.2, 76.2]
# print(a11.index(max(a11)), max(a11))
# a13=[56.0, 62.0, 63.48888888888889, 66.575, 70.04444444444445, 64.18181818181819, 69.84210526315789, 69.53846153846153, 64.08333333333333, 71.66666666666667, 73.66666666666667, 75.0, 74.33333333333333, 76.33333333333333, 76.6, 75.78947368421052, 74.64444444444445, 76.66666666666667, 75.36842105263158, 75.4, 73.91304347826087, 74.0, 76.0, 72.56521739130434, 73.52631578947368, 77.3157894736842, 72.63636363636364, 75.0, 78.10526315789474, 78.25, 73.15789473684211, 77.5, 75.73684210526316, 78.78947368421052, 77.63157894736842, 74.225, 78.3157894736842, 74.54285714285714, 77.84210526315789, 76.22857142857143, 77.48571428571428, 76.28571428571429, 77.86363636363636, 78.0, 77.02857142857142, 78.21052631578948, 77.18181818181819, 77.13636363636364, 77.54545454545455, 77.0909090909091, 78.0909090909091, 77.5909090909091, 77.34285714285714, 77.81818181818181, 76.88571428571429, 77.33333333333333, 77.84210526315789, 77.45454545454545, 77.45454545454545, 77.36363636363636]
# print(a13.index(max(a13)), max(a13))
# a14=[56.0, 59.0, 51.78947368421053, 68.44444444444444, 70.66666666666667, 70.28888888888889, 69.57894736842105, 70.44444444444444, 72.0, 71.7, 72.13333333333334, 72.55555555555556, 70.47368421052632, 75.04444444444445, 71.2, 74.77777777777777, 72.0, 72.74285714285715, 76.33333333333333, 76.66666666666667, 69.2439024390244, 75.0, 77.15789473684211, 69.62857142857143, 73.33333333333333, 75.075, 76.0, 73.66666666666667, 76.35555555555555, 77.74193548387096, 77.66666666666667, 75.84210526315789, 78.33333333333333, 78.33333333333333, 78.2, 78.36842105263158, 78.78947368421052, 76.04347826086956, 79.33333333333333, 78.33333333333333, 79.33333333333333, 76.0, 78.33333333333333, 78.66666666666667, 74.66666666666667, 80.66666666666667, 77.66666666666667, 77.44444444444444, 79.33333333333333, 80.5, 78.27272727272727, 81.0]
# print(a14.index(max(a14)), max(a14))
# a15=[52.86666666666667, 55.733333333333334, 51.2, 68.44444444444444, 70.33333333333333, 70.28888888888889, 69.44444444444444, 69.77777777777777, 71.11111111111111, 71.4888888888889, 72.13333333333334, 72.55555555555556, 70.37777777777778, 75.04444444444445, 71.04444444444445, 74.77777777777777, 70.75555555555556, 72.5111111111111, 75.33333333333333, 76.17777777777778, 69.0, 73.46666666666667, 76.77777777777777, 69.5111111111111, 72.4, 74.95555555555555, 75.15555555555555, 72.73333333333333, 76.35555555555555, 77.46666666666667, 77.33333333333333, 75.5111111111111, 76.55555555555556, 77.82222222222222, 77.44444444444444, 77.4888888888889, 77.84444444444445, 75.73333333333333, 77.88888888888889, 77.44444444444444, 78.64444444444445, 74.73333333333333, 76.46666666666667, 76.42222222222222, 74.08888888888889, 78.13333333333334, 76.4, 76.4888888888889, 78.4888888888889, 77.02222222222223, 77.6, 79.11111111111111, 79.8, 77.31111111111112, 74.82222222222222, 77.6, 78.77777777777777, 76.22222222222223, 80.15555555555555, 79.44444444444444, 78.2, 78.31111111111112, 77.53333333333333, 78.31111111111112, 78.73333333333333, 78.66666666666667, 76.86666666666666, 79.57777777777778, 75.88888888888889, 77.64444444444445, 78.15555555555555, 76.84444444444445, 76.84444444444445, 77.33333333333333, 76.64444444444445, 77.4888888888889, 78.75555555555556, 77.97777777777777, 78.0, 78.0, 78.35555555555555, 77.06666666666666, 77.35555555555555, 78.86666666666666, 77.97777777777777, 78.04444444444445, 78.84444444444445, 78.77777777777777, 77.06666666666666, 78.46666666666667, 78.42222222222222, 78.86666666666666, 79.11111111111111, 77.71111111111111, 79.13333333333334, 78.4888888888889, 78.5111111111111, 78.28888888888889, 78.42222222222222, 78.42222222222222]
# print(a15.index(max(a15)), max(a15))
# # plt.plot([i for i in range(1, 101)], a, label='Valid')
# # plt.savefig('_Accuracy.png')

# import os

# # Define the paths to the two folders you want to compare
# folder1_path = '/home/Student/s4737925/Project/Patient_Split/test/AD'
# folder2_path = '/home/Student/s4737925/Project/Patient_Split/valid/AD'

# # Get the list of files in each folder
# files_in_folder1 = os.listdir(folder1_path)
# files_in_folder2 = os.listdir(folder2_path)

# # Convert the lists to sets for easy comparison
# set1 = set(files_in_folder1)
# set2 = set(files_in_folder2)

# print(set1,set2)
# # Check if the sets are equal, meaning the folders have the same files
# if set1 == set2:
#     print("The folders have the same files.")
# else:
#     # Find the files that are unique to each folder
#     unique_to_folder1 = set1 - set2
#     unique_to_folder2 = set2 - set1

#     print("The folders do not have the same files.")
#     print("Files unique to folder 1:", unique_to_folder1)
#     print("Files unique to folder 2:", unique_to_folder2)


# # python code to load and visualize 
# # an image

# # import necessary libraries
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np

# # load the image
# img_path = '218391_79.jpeg'
# img = Image.open(img_path)

# # convert PIL image to numpy array
# img_np = np.array(img)

# # plot the pixel values
# plt.hist(img_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")
# #plt.savefig('_Batch.png')

# # Python code for converting PIL Image to
# # PyTorch Tensor image and plot pixel values

# # import necessary libraries
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# # define custom transform function
# transform = transforms.Compose([
# 	transforms.ToTensor()
# ])

# # transform the pIL image to tensor 
# # image
# img_tr = transform(img)

# # Convert tensor image to numpy array
# img_np = np.array(img_tr)

# # plot the pixel values
# plt.hist(img_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")

# # Python code to calculate mean and std
# # of image

# # get tensor image
# img_tr = transform(img)

# # calculate mean and std
# mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
# mean=(0.1156)
# std=(0.1295)
# # print mean and std
# print("mean and std before normalize:")
# print("Mean of the image:", mean)
# print("Std of the image:", std)

# # python code to normalize the image


# from torchvision import transforms

# # define custom transform
# # here we are using our calculated
# # mean & std
# transform_norm = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize(mean, std)
# ])

# # get normalized image
# img_normalized = transform_norm(img)

# # convert normalized image to numpy
# # array
# img_np = np.array(img_normalized)

# # plot the pixel values
# plt.hist(img_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")
# plt.savefig('Batch11.png')

# # Python Code to visualize normalized image

# # get normalized image
# img_normalized = transform_norm(img)

# # convert this image to numpy array
# img_normalized = np.array(img_normalized)

# # transpose from shape of (3,,) to shape of (,,3)
# img_normalized = img_normalized.transpose(1, 2, 0)

# # display the normalized image
# plt.imshow(img_normalized)
# plt.xticks([])
# plt.yticks([])
# plt.savefig('Batch00.png')

# img_nor = transform_norm(img)
 
# # cailculate mean and std
# mean, std = img_nor.mean([1,2]), img_nor.std([1,2])
 
# # print mean and std
# print("Mean and Std of normalized image:")
# print("Mean of the image:", mean)
# print("Std of the image:", std)

# # Normalize
# def get_mean_std(loader):
#     mean = 0.
#     std = 0.
#     for images, _ in loader:
#         batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
#         images = images.view(batch_samples, images.size(1), -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)

#     mean /= len(loader.dataset)
#     std /= len(loader.dataset)
#     var = 0.0
#     pixel_count = 0
#     for images, _ in loader:
#         batch_samples = images.size(0)
#         images = images.view(batch_samples, images.size(1), -1)
#         var += ((images - mean.unsqueeze(1))**2).sum([0,2])
#         pixel_count += images.nelement()
#     std1 = torch.sqrt(var / pixel_count)
#     print(mean, std, std1)
#     return mean, std, std1

# def normalize_train():
#     transform_train = transforms.Compose([
#     transforms.Resize((size,size)),
#     transforms.ToTensor(),
# ])
#     trainset = torchvision.datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/train', transform=transform_train)
#     #torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
#     #trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)   
#     mean, std, std1 = get_mean_std(trainloader) 
#     return mean,std, std1

# def normalize_test():
#     transform_test = transforms.Compose([
#     transforms.Resize((size,size)),
#     transforms.ToTensor(),
# ])
#     testset = torchvision.datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/test', transform=transform_test)
#     #torchvision.datasets.ImageFolder(root='/home/Student/s4737925/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
#     #trainset = torchvision.datasets.ImageFolder(root='Z:/Project/Dataset/ADNI_AD_NC_2D/AD_NC/train', transform=transform_train)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)   
#     mean, std, std1 = get_mean_std(testloader) 
#     return mean, std, std1  