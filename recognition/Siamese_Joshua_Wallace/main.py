import torch
from modules import Siamese
from utils import Config
from dataset import Dataset
from train import Train

if __name__ == '__main__':
    net = Siamese()
    config = Config()
    dataset = Dataset()
    trainer = Train(net, dataset, config)
    
    trainer.train()
    trainer.test()
    