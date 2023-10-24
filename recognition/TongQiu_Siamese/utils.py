import torch
from pathlib import Path


class Config:

    # # run on local machine
    # TRAIN_DIR = Path("/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/train")
    # TEST_DIR = Path("/Users/tongqiu/Desktop/COMP3710_Report/AD_NC/test")
    # DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    # LOG_DIR = Path("/Users/tongqiu/Desktop/COMP3710_Report/log")
    # MODEL_DIR_CONTRASTIVE = "/Users/tongqiu/Desktop/COMP3710_Report/Contrastive.pth"
    # MODEL_DIR_TRIPLET = "/Users/tongqiu/Desktop/COMP3710_Report/Triplet.pth"

    # run on Rangpur
    TRAIN_DIR = Path("./AD_NC/train")
    TEST_DIR = Path("./AD_NC/test")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_DIR = Path("./log")
    MODEL_DIR_CONTRASTIVE = "./model/Contrastive.pth"
    MODEL_DIR_TRIPLET = "./model/Triplet.pth"
    MODEL_DIR_CLASSIFIER = "./model/Classifier_cf.pth"

    # # colab
    # TRAIN_DIR = Path("/content/drive/MyDrive/Colab_Notebooks/COMP_3710/Siamese/AD_NC/train")
    # TEST_DIR = Path("/content/drive/MyDrive/Colab_Notebooks/COMP_3710/Siamese/AD_NC/test")
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # LOG_DIR = Path("/content/drive/MyDrive/Colab_Notebooks/COMP_3710/Siamese/log/Contrastive_0")
    # MODEL_DIR = "/content/drive/MyDrive/Colab_Notebooks/COMP_3710/Siamese/Contrastive_0.pth"




