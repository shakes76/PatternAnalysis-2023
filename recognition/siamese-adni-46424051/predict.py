##################################   predict.py   ##################################

from train import Train

class Predict():
    def __init__(self):
        self.train = Train()
        self.train.train()
        self.train.test()