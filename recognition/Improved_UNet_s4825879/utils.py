from torchvision import transforms
import random

class DataTransform(object):
    def __init__(self, p=0.5, size=(512,512)):
        super(DataTransform, self).__init__()
        self.p = p
        self.size = size
       
    def __call__(self, input):
        img = input[0]
        truth = input[1]
        
        if random.randint(0,1) < self.p:
           img = transforms.functional.vflip(img)
           truth = transforms.functional.vflip(truth)

        if random.randint(0,1) < self.p:
            img = transforms.functional.hflip(img)
            truth = transforms.functional.hflip(truth)
            
        integer = random.randint(0,1)
        if integer < self.p:
           img = transforms.functional.rotate(img, integer*365)
           truth = transforms.functional.rotate(truth, integer*365)

        img = transforms.functional.to_tensor(img)
        truth = transforms.functional.to_tensor(truth)
        
        img = transforms.functional.resize(img, self.size)
        truth = transforms.functional.resize(truth, self.size)

        return(img, truth)
