from torchvision import transforms
import torch
import random
class RandomTransformTwoElements:
    def __init__(self, transform,transform_seg):
        self.transform = transform
        self.transform_seg=transform_seg

    def __call__(self, element1, element2):
        # Apply the same random transformation to both elements
        seed=random.randint(0, 1000)
        random.seed(seed) 
        torch.manual_seed(seed)
        output1=self.transform(element1)
        random.seed(seed) 
        torch.manual_seed(seed)
        output2=self.transform_seg(element2)
        return (output1,output2)

#transform_norm = transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512),antialias=True)])