import torch


class Config() :
    def __init__(self, 
            lr=1e-3, 
            wd=1e-5, 
            epochs=10, 
            batch_size=32, 
            root_dir='./AD_CN',
            gpu = 'cuda'
    ) -> None :
        
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.root_dir = root_dir

        if gpu == 'cuda' :
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif gpu == 'mps' :
            self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        else :
            self.device = torch.device('cpu')
        