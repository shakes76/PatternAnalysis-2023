import torch
from pathlib import Path


class Config:
    TRAIN_DIR = Path("/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/train")
    TEST_DIR = Path("/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/test")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_DIR = Path("/Users/tongqiu/Desktop/COMP3710_Report/log/SimpleCNN")
    MODEL_DIR = '/Users/tongqiu/Desktop/COMP3710_Report/model.pth'

