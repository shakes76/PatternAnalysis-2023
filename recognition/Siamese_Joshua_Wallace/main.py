import torch
from modules import Siamese
from utils import Config
from dataset import get_dataloaders
from train import Train

if __name__ == '__main__':
    net = Siamese()
    train_loader, test_loader = get_dataloaders()
    config = Config()
    trainer = Train(net, train_loader, config)
    trainer.train()
    