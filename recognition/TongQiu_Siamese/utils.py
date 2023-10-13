import torch


class Config:
    TRAIN_DIR = "/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/train"
    TEST_DIR = "/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
