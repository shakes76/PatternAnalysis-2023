##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
import matplotlib.pyplot as plt
import numpy as np

class Predict() :
    def __init__(self, input, n = 16, savepath = './', model = None, path = None) :
        self.savepath = savepath
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None and path is not None :
            model = torch.load(path)
            self.model = model.to(self.device)
        else :
            self.model = model

        self.model.eval()
        self.output = None
        self.input = input
        self.n = n

    def generate(self) :
        with torch.no_grad() :
            output = self.model(self.input)
            self.output = output.cpu().numpy()
    
    def show_generated(self, save = True) :
        if self.output is None :
            self.generate()
            
        for i in range(self.n):
            plt.imshow(np.transpose(self.output[i], (1, 2, 0)))
            plt.axis('off')
            if save :
                plt.savefig(self.savepath + '_generated.png')
            else :
                plt.show()


class SSIM():
    def __init__() :
        pass