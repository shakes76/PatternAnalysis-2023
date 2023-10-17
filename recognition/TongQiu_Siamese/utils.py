import torch
from pathlib import Path


class Config:
    TRAIN_DIR = Path("~/comp3710/report/AD_NC/train")  # Path("/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/train")
    TEST_DIR = Path("~/comp3710/report/AD_NC/rest")  # Path("/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/test")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    LOG_DIR = Path("~/comp3710/report/log/Contrastive_0")  # Path("/Users/tongqiu/Desktop/COMP3710_Report/log/Contrastive_0")
    MODEL_DIR = '~/comp3710/report/model/Contrastive_0.pth'  # '/Users/tongqiu/Desktop/COMP3710_Report/Contrastive_0.pth'



