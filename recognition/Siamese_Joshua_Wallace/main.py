import torch
from modules import Siamese
from utils import Config
from dataset import ADNIDataset
from train import Train

if __name__ == '__main__':
    net = Siamese()
    config = Config(lr=1e-4, wd=1e-5, epochs=1, batch_size=32, root_dir='./AD_NC', gpu='cuda')
    dataset = ADNIDataset(batch_size=config.batch_size, fraction=0.1)
    print('Dataset loaded, beginning training.')
    trainer = Train(net, dataset, config)
    trainer.train()
    print('Model trained, beginning testing.')
    trainer.test()
    trainer.plot_loss()