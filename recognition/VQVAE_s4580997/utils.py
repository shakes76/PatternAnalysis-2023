import torch


class Config() :
    def __init__(self) :
        
        self.lr = 1e-4
        self.wd = 1e-5
        self.epochs = 1
        self.batch_size = 32
        self.root_dir = './AD_CN'
        self.savepath = './models/vqvae'
        gpu = 'cuda'

        if gpu == 'cuda' :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif gpu == 'mps' :
            self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        else :
            self.device = torch.device('cpu')
        