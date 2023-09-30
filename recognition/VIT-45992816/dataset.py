from utils import utilis_augs
import torchvision, torch
import numpy as np
import os
from utils.utilis_ import Metrice_Dataset

def get_transform(opt):
    # get train and validation transformation
    train_transform, val_transform = utilis_augs.get_dataprocessing(torchvision.datasets.ImageFolder(opt.train_path),
                                 opt)
    return train_transform, val_transform

def get_test_transform(train_opt, opt):
    # get test transformation
    test_transform = utilis_augs.get_dataprocessing_teststage(train_opt, opt,
                                             torch.load(os.path.join(opt.save_path, 'preprocess.transforms')))
    return test_transform


def preprocessing_train_loader(opt,train_transform):
    # load train data
    train_data = torchvision.datasets.ImageFolder(opt.train_path, transform=train_transform)
    batch_size = opt.batch_size
    class_weight = np.array((2,1))#np.ones_like(np.unique(train_data.targets))
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=opt.workers)
    return train_dataset,class_weight

def preprocessing_val_loader(opt,val_transform):
    # load val data
    val_data = torchvision.datasets.ImageFolder(opt.val_path, transform=val_transform)
    batch_size = opt.batch_size
    val_dataset = torch.utils.data.DataLoader(val_data, max(batch_size // 1, 1),
                                               shuffle=False, num_workers=opt.workers)

    return val_dataset

def preprocessing_test_loader(opt,test_transform):
    # load test data
    test_dataset = Metrice_Dataset(
        torchvision.datasets.ImageFolder(eval('opt.{}_path'.format(opt.task)), transform=test_transform))
    test_dataset = torch.utils.data.DataLoader(test_dataset, opt.batch_size, shuffle=False,
                                               num_workers=opt.workers)
    return test_dataset