import torch
from modules import Siamese
from utils import Config
from dataset import Dataset
from train import Train

if __name__ == '__main__':
    net = Siamese()
    # Limit batch size, else the program will run out of memory
    config = Config(lr=1e-3, wd=1e-5, epochs=1, batch_size=32, root_dir='./AD_NC', gpu='cuda')
    dataset = Dataset(batch_size=config.batch_size)
    trainer = Train(net, dataset, config)
    trainer.train()
    trainer.test()
    