import torch
from modules import Siamese
from utils import Config
from dataset import Dataset
from train import Train

if __name__ == '__main__':
    net = Siamese()
    dataset = Dataset()
    dataset.load_train()
    config = Config()
    trainer = Train(net, dataset.get_train(), config)
    trainer.train()
    