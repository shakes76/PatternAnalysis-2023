import torch
import matplotlib.pyplot as plt
import numpy as np

class Predict() :
    def __init__(self, path, input, n = 16, savepath = './') :
        self.path = path
        self.savepath = savepath
        model = torch.load(self.path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.output = None
        self.input = input
        self.n = n

    def generate(self) :
        with torch.no_grad() :
            output = self.model(self.input)
            output = output.cpu().numpy()
    
    def show_generated(self, save = True) :
        for i in range(self.n):
            plt.imshow(np.transpose(self.output[i], (1, 2, 0)))
            plt.axis('off')
            if save :
                plt.savefig(self.savepath + '_generated.png')
            else :
                plt.show()
