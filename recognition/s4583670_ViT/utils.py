'''
@file   utils.py
@brief  Contains the source code for utility functions
@date   20/10/2023
'''

import os
import random
from collections import defaultdict
from shutil import move

'''
Reorganise the training file for a training/validation patient-level split and to meet a specified
split ratio
Parameters:
    directory - training file directory
    valid_dir - new validation file directory
    file_limit - number of files to store in validation directory
'''
def create_train_val_split(directory, valid_dir):
    files = defaultdict(set)
    file_limit = int(len(os.listdir(directory)) * 0.2)
    print(file_limit)
    
    for file in os.listdir(directory):
        key = int(file.split("_")[0])
        files[key].add(file)
    
    keys = list(files.keys())
    random.shuffle(keys)
    
    file_count = 0
    for key in keys:
        file_set = files[key]
        for file in file_set:
            src_path = os.path.join(directory, file)
            dest_path = os.path.join(valid_dir, file)
            move(src_path, dest_path)
            file_count += 1
        if file_count >= file_limit:
            break

if __name__ == "__main__":
    data_dir = "AD_NC/"

    # Make the validation directorys
    os.makedirs(os.path.join(data_dir, "valid/AD"))
    os.makedirs(os.path.join(data_dir, "valid/NC"))

    train_AD_dir = os.path.join(data_dir, "train/AD")
    train_NC_dir = os.path.join(data_dir, "train/NC")

    valid_AD_dir = os.path.join(data_dir, "valid/AD")
    valid_NC_dir = os.path.join(data_dir, "valid/NC")

    # Move the files
    create_train_val_split(train_AD_dir, valid_AD_dir)
    create_train_val_split(train_NC_dir, valid_NC_dir)
