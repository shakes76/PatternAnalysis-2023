from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
# import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import shutil
import random
from Modules import UNet_For_Brain
from Losses import calc_loss, dice_loss, threshold_predictions_v, threshold_predictions_p
from Ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow, draw_loss
from Metrics import dice_coeff, accuracy_score
import time

# from ploting import VisdomLinePlotter
# from visdom import Visdom


#######################################################
# Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
# Setting the basic paramters of the model
#######################################################

batch_size = 16
print('batch_size = ' + str(batch_size))

valid_size = 0.15

epoch = 400
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 24
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch - 2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

# plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
# Setting up the model
#######################################################

model_Inputs = [UNet_For_Brain]


def model_unet(model_input, in_channel=1, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test


# passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary


model_test = model_unet(model_Inputs[-1], 3, 1)

model_test.to(device)

#######################################################
# Getting the Summary of Model
#######################################################

torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
# Passing the Dataset of Images and Labels
#######################################################


#ISIC2018 data
t_data = './ISIC2018/ISIC2018_Task1-2_Training_Input_x2/'
l_data = './ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/'
test_image = './keras_png_slices_data/keras_png_slices_test/'
test_label = './keras_png_slices_data/keras_png_slices_seg_test/'
test_folderP = './ISIC2018/ISIC2018_Task1-2_Training_Input_x2/*'
test_folderL = './ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/*'
valid_image = './ISIC2018/ISIC2018_Task1-2_Training_Input_x2/'
valid_lable = './ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2/'

Training_Data = Images_Dataset_folder(t_data, l_data)

Validing_Data = Images_Dataset_folder(valid_image, valid_lable)

#######################################################
# Giving a transformation for input data
#######################################################

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    #   torchvision.transforms.CenterCrop(96),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#######################################################
# Trainging Validation Split
#######################################################

num_train = len(Training_Data)
indices_train = list(range(num_train))
# split = int(np.floor(valid_size * num_train))

num_valid = len(Validing_Data)
indices_valid = list(range(num_valid))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices_train)
    np.random.shuffle(indices_valid)

# train_idx, valid_idx = indices[split:], indices[:split]
train_idx, valid_idx = indices_train, indices_valid
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory, )

valid_loader = torch.utils.data.DataLoader(Validing_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory, )

#######################################################
# Using Adam as Optimizer
#######################################################
# Hyper Parameters
initial_lr = 5e-4
lr_decay = 0.985
l2_weight_decay = 1e-5
# initial_lr = 0.001
# opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
# # opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

# MAX_STEP = int(1e10)
# eta_min = 1e-2
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)

# Optimisation & Loss Settings
opt = optim.Adam(model_test.parameters(), lr=initial_lr, weight_decay=l2_weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=lr_decay)

New_folder = './model'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
# Setting the folder of saving the predictions
#######################################################

read_pred = './model/pred'

#######################################################
# Checking if prediction folder exixts
#######################################################

if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

#######################################################
# checking if the model exists and if true then delete
#######################################################

read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

#######################################################
# Training loop
#######################################################

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    train_loss_list = []
    valid_loss_list = []
    since = time.time()
    scheduler.step()
    # lr = scheduler.get_lr()

    #######################################################
    # Training Data
    #######################################################

    model_test.train()
    k = 1

    for x, y in train_loader:
        # print("x: ", x.shape)
        # print("y: ", y.shape)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()

        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)  # Dice_loss Used

        train_loss += lossT.item() * x.size(0)
        lossT.backward()
        opt.step()
        x_size = lossT.item() * x.size(0)
        k = 2

    train_loss_list.append(train_loss)

    #######################################################
    # Validation Step
    #######################################################

    model_test.eval()
    torch.no_grad()  # to increase the validation process uses less memory

    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)

        y_pred1 = model_test(x1)
        lossL = calc_loss(y_pred1, y1)  # Dice_loss Used

        valid_loss += lossL.item() * x1.size(0)
        x_size1 = lossL.item() * x1.size(0)

    valid_loss_list.append(valid_loss)

    if (i + 1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
    #######################################################
    # Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min and epoch_valid >= i:  # and i_valid <= 2:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(), './model/Unet_D_' +
                   str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth')

        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid + 1
        valid_loss_min = valid_loss

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#######################################################
# Loading the model
#######################################################

model_test.load_state_dict(torch.load('./model/Unet_D_' +
                                      str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                      + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()

#######################################################
# opening the test folder and creating a folder for generated images
#######################################################

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort

read_test_folder112 = './model/gen_images'

if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)

# For Prediction Threshold

read_test_folder_P_Thres = './model/pred_threshold'

if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

# For Label Threshold

read_test_folder_L_Thres = './model/label_threshold'

if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)

#######################################################
# saving the images in the files
#######################################################

img_test_no = 0

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i]).convert("RGB")

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)

    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255

    s = data_transform(im)
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = F.sigmoid(pred)
    pred = pred.detach().numpy()

    #    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

    if i % 24 == 0:
        img_test_no = img_test_no + 1

    x1 = plt.imsave('./model/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', pred[0][0], cmap='gray')

####################################################
# Calculating the Dice Score
####################################################

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    #    torchvision.transforms.CenterCrop(96),
    torchvision.transforms.Grayscale(),
    #            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

read_test_folderP = glob.glob('./model/gen_images/*')
x_sort_testP = natsort.natsorted(read_test_folderP)

read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort

dice_score123 = 0.0
x_count = 0
x_dice = 0

for i in range(len(read_test_folderP)):

    x = Image.open(x_sort_testP[i]).convert("L")
    s = data_transform(x)
    s = np.array(s)
    s = threshold_predictions_v(s)
    # print(s)
    # print("------------------")

    # save the images
    x1 = plt.imsave('./model/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s)

    y = Image.open(x_sort_testL[i]).convert("L")
    s2 = data_transform(y)
    s3 = np.array(s2)
    s2 = threshold_predictions_v(s2)
    # print(s3)

    # save the Images
    y1 = plt.imsave('./model/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s3)

    total = dice_coeff(s, s3)
    print(total)

    if total <= 0.8:
        x_count += 1
    if total > 0.8:
        x_dice = x_dice + total
    dice_score123 = dice_score123 + total

# print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))
print(x_count)
print(x_dice)
print('Dice Score : ' + str(float(x_dice / (len(read_test_folderP) - x_count))))

