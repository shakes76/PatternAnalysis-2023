import torch


class VQVAEConfig() :
    def __init__(self) :
        self.lr = 1e-3
        self.wd = 1e-5
        self.epochs = 5
        self.batch_size = 32
        self.root_dir = './AD_CN'
        self.savepath = './models/vqvae'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GANConfig() :
    def __init__(self) :
        self.lr = 1e-3
        self.wd = 1e-5
        self.epochs = 1
        self.batch_size = 32
        self.root_dir = './AD_CN'
        self.savepath = './models/gan'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        