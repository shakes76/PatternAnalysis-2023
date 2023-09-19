import torch


class Config() :
    def __init__(self, 
            lr=1e-3, 
            wd=1e-5, 
            epochs=10, 
            batch_size=32, 
            root_dir='./AD_CN') :
        
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.root_dir = root_dir
        