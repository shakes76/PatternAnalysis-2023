from torchvision import transforms
import torch
import random
class RandomTransform:
    def __init__(self, transform,transform_seg):
        self.transform = transform          #Transform for normal images    
        self.transform_seg=transform_seg    #Transform for turth

    def __call__(self, element1, element2):
        seed=random.randint(0, 1000) #Create a random seed

        random.seed(seed)                   #Set random seed
        torch.manual_seed(seed)             #Set random seed for torch
        output1=self.transform(element1)    #Transform for normal images

        random.seed(seed)                   #Resets seed so the random transforms are the same for truth and normal
        torch.manual_seed(seed)
        output2=self.transform_seg(element2)

        return (output1,output2)

