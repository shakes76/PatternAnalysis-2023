"""
Dataset loader
"""
import torch
import os

from PIL import Image

class Data():
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.AD, self.NC = None, None
        self.loader()
    
    def loader(self):
        self.AD = self.loadPath("AD")
        self.NC = self.loadPath("NC")

    def loadPath(self, type):
        arr = []
        for imageName in  os.listdir(os.path.join(self.dataPath, type)):
            curImage = Image.open(os.path.join(self.dataPath, type, imageName)).convert("L")
            arr.append(curImage)
        return arr
    
    def testLen(self):
        return (len(self.AD), len(self.NC))
    
if __name__ == "__main__":
    test = Data("AD_NC/test")
    train = Data("AD_NC/train")
    print(train.testLen())
    print(test.testLen())
