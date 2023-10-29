from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data(data, log_res, batchSize):

    transform = transforms.Compose(
        [   transforms.Resize(size=(2**log_res, 2**log_res), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

    dataset = datasets.ImageFolder(root=data, transform=transform)

    loader = DataLoader(dataset, batchSize, shuffle=True)
    

    return loader